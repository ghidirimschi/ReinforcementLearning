import torchvision as torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
import data_augs
from collections import deque

total_steps = 32e6
num_envs = 32
num_levels = 200 # 10
num_steps = 256
num_epochs = 3
batch_size = 128 #8 #512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
aug_coef = 0.1
ucb_exploration_coef = 0.5
total_num = 1
ucb_window_length = 10
num_interval = num_envs * num_steps
grayscale = False

feature_dim = 64
# Clamp function just in case:
def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

aug_to_func = {
        # 'crop': data_augs.Crop,
        'grayscale': data_augs.Grayscale,
        'random-conv': data_augs.RandomConv,
        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        'cutout-color': data_augs.CutoutColor,
        'color-jitter': data_augs.ColorJitter,
}

number_of_augs = len(aug_to_func)
expl_action = [0.] * number_of_augs
ucb_action = [0.] * number_of_augs
num_action = [1.] * number_of_augs
qval_action = [0.] * number_of_augs
return_action = []
for i in range(number_of_augs):
            return_action.append(deque(maxlen=ucb_window_length))

aug_id = data_augs.Identity
aug_list = [aug_to_func[t](batch_size=batch_size) for t in list(aug_to_func.keys())]

def select_ucb_aug():
    for i in range(number_of_augs):
        expl_action[i] = ucb_exploration_coef * \
            np.sqrt(np.log(total_num) / num_action[i])
        ucb_action[i] = qval_action[i] + expl_action[i]
    print(expl_action)
    print(qval_action)
    print(ucb_action)
    ucb_aug_id = np.argmax(ucb_action)
    return ucb_aug_id, aug_list[ucb_aug_id]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, in_channels, depth):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return x + out


class ConvSequence(nn.Module):
    def __init__(self, in_channels, depth):
        super(ConvSequence, self).__init__()
        self.conv = nn.Conv2d(in_channels, depth, 3, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.residual1 = Residual(depth, depth)
        self.residual2 = Residual(depth, depth)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxPool(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            ConvSequence(1 if grayscale else in_channels, 16),
            ConvSequence(16, 32),
            ConvSequence(32, 32),
            Flatten(),
            nn.ReLU(),
            nn.Linear(1568, feature_dim)
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        if grayscale:
            x = torchvision.transforms.Grayscale()(x)
        return self.layers(x)


class Policy(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            value, dist, _ = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return value, dist, logits

    def get_value(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            value, _, _ = self.forward(x)

        return value

if __name__ == "__main__":
    # Define environment
    # check the utils.py file for info on arguments
    env = make_env(num_envs, num_levels=num_levels)
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space.n)

    # Define network

    input_channels = env.observation_space.shape[0]
    # print("Input channels: ", input_channels, env.observation_space.shape[1], env.observation_space.shape[2])
    # exit(0)
    encoder = Encoder(input_channels, feature_dim)
    policy = Policy(encoder, feature_dim, env.action_space.n)
    policy.cuda()

    # Define optimizer
    # these are reasonable values but probably not optimal
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

    # Define temporary storage
    # we use this to collect transitions during each iteration
    storage = Storage(
        env.observation_space.shape,
        num_steps,
        num_envs
    )

    # Run training
    obs = env.reset()
    step = 0
    max_mean = 0
    while step < total_steps:

        # open a file by creating it as text
        f = open('data5.txt','a')
        
        # Use policy to collect data for num_steps steps
        policy.eval()
        for i in range(num_steps):

            # Use policy
            action, log_prob, value = policy.act(obs)

            # Take step in environment
            next_obs, reward, done, info = env.step(action)

            # Store data
            storage.store(next_obs, action, reward, done, info, log_prob, value)

            # Update current observation
            obs = aug_id(storage.obs[i])

        # Add the last observation to collected data
        obs = aug_id(storage.obs[-1])
        _, _, value = policy.act(obs)
        storage.store_last(obs, value)

        # Compute return and advantage
        storage.compute_return_advantage()

        # Optimize policy
        policy.train()

        # Update UCB values
        if step > 0:
          total_num += num_interval
          num_action[current_aug_id] += num_interval
          return_action[current_aug_id].append(storage.returns.mean().item())
          qval_action[current_aug_id] = np.mean(return_action[current_aug_id])

        # Select an augmentation
        current_aug_id, current_aug_func = select_ucb_aug() 
        print(current_aug_id)
        print(current_aug_func)

        for epoch in range(num_epochs):

            # Iterate over batches of transitions
            generator = storage.get_generator(batch_size)
            for batch in generator:
                b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

                # Get current policy outputs
                new_value, new_dist, _ = policy(b_obs)
                new_log_prob = new_dist.log_prob(b_action)

                # Clipped policy objective
                # pi_loss = self.pi_loss(new_log_prob, b_log_prob, b_advantage, eps)
                ratio = torch.exp(new_log_prob - b_log_prob)
                clipped_ratio = ratio.clamp(min=1.0 - eps, max=1.0 + eps)
                policy_reward = torch.min(ratio * b_advantage, clipped_ratio * b_advantage)
                pi_loss = -policy_reward.mean()

                # Clipped value function objective
                # value_loss = self.value_loss(new_value, b_value, b_returns, eps)
                clipped_value = b_value + (new_value - b_value).clamp(min=-eps, max=eps)
                vf_loss = torch.max((new_value - b_returns) ** 2, (clipped_value - b_returns) ** 2)
                value_loss = 0.5 * vf_loss.mean()

                # Entropy loss
                entropy_loss = new_dist.entropy()
                entropy_loss = entropy_loss.mean()

                # Augmentation loss
                b_obs_aug = current_aug_func.do_augmentation(b_obs)
                b_obs_id = aug_id(b_obs)
                value_aug, dist_aug, _ = policy(b_obs_aug)
                action_loss_aug = - dist_aug.log_prob(dist_aug.sample()).mean()
                value_loss_aug = 0.5 * ((torch.detach(new_value) - value_aug) ** 2).mean()
               
                aug_loss = value_loss_aug + action_loss_aug

                # Backpropagate losses
                # loss = (pi_loss + value_coef * value_loss - entropy_coef * entropy_loss)
                loss = (pi_loss + value_coef * value_loss - entropy_coef * entropy_loss + aug_loss * aug_coef)
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

                # Update policy
                optimizer.step()
                optimizer.zero_grad()

                current_aug_func.change_randomization_params_all()

        # Update stats
        step += num_interval
        print(f'Step: {step}\tMean reward: {storage.get_reward()}')
        # write to the file
        f.write(f'Step: {step}\tMean reward: {storage.get_reward()}\n')
        # close the file
        f.close()
        if storage.get_reward() > max_mean:
            print('New high mean. Updating...')
            torch.save(policy.state_dict(), 'checkpoint5.pt')
            max_mean = storage.get_reward()

    print('Completed training!')
