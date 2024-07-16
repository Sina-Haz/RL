# In this file we will train the lunar lander for unit 1

import argparse
import gymnasium as gym
from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Set up argument parser
parser = argparse.ArgumentParser(description="Train a PPO model on the LunarLander environment")
parser.add_argument('--timesteps', type=int, default=100, help='Number of training timesteps')
parser.add_argument('--save_path', type=str, default='models/ppo-Lander-default', help='Filename to save the model')

# Parse arguments
args = parser.parse_args()

# Create vectorized trainining environment
env = make_vec_env('LunarLander-v2', n_envs=16)

# Load previous model from huggingface so not starting from scratch
checkpoint = load_from_hub(
	repo_id="shazeghi/ppo-LunarLander-v2",
	filename="ppo-LunarLander-v2.zip",
)

# load in model with set training environment
model = PPO.load(checkpoint, env=env)

# Ensure model is using GPU:
print(model.device)

# Train for however many timesteps you like
model.learn(total_timesteps = args.timesteps)

# create evaluation environment and print model evaluation
# We wrap the environment in a monitor so it can be displayed
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))

# Stable-Baselines3 provides function evaluate_policy to check how well our model performs
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Lastly we would save our model
model.save(args.save_path)
