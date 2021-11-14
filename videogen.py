import imageio
import torch
from torch import nn

from main import num_envs, num_levels, Encoder, feature_dim, Policy, grayscale, Flatten
from utils import make_env
# Make evaluation environment
# class Flatten(nn.Module):
#   def forward(self, x):
#     return x.view(x.size(0), -1)

eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels) #added distribution_mode
obs = eval_env.reset()

frames = []
total_reward = []

# Load policy:
input_channels = eval_env.observation_space.shape[0]
encoder = Encoder(input_channels, feature_dim)
policy = Policy(encoder, feature_dim, eval_env.action_space.n)
policy.cuda()
policy.load_state_dict(torch.load('checkpoint.pt'))
# Evaluate policy:
policy.eval()

print("Generating video...")
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid.mp4', frames, fps=30)
print('Video saved')