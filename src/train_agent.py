import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Optional, Dict

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.target = np.array([0.8, 0.8, 0, 0])
        self.state = None
        self.steps = 0
        self.max_steps = 100
        self.prev_distance = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-0.5, 0.5, size=4)
        self.steps = 0
        self.prev_distance = np.linalg.norm(self.state - self.target)
        return self.state.copy(), {}

    def step(self, action):
        action_effects = np.array([
            [0.15, 0, 0, 0],   # Right
            [-0.15, 0, 0, 0],  # Left
            [0, 0.15, 0, 0],   # Up
            [0, -0.15, 0, 0]    # Down
        ])
        
        self.state = np.clip(self.state + action_effects[action], -1, 1)
        current_distance = np.linalg.norm(self.state - self.target)
        
       
        proximity_reward = 15 * (1 - current_distance)
        movement_reward = 20 * (self.prev_distance - current_distance)
        reward = proximity_reward + movement_reward
        self.prev_distance = current_distance
        
        self.steps += 1
        done = current_distance < 0.1 or self.steps >= self.max_steps
        
        return self.state.copy(), reward, done, False, {}

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        return True


os.makedirs("logs", exist_ok=True)
env = CustomEnv()
env = Monitor(env, "logs")
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={
        'net_arch': [dict(pi=[128, 128], vf=[128, 128])]},
    verbose=1
)


callback = RewardCallback()
model.learn(total_timesteps=100000, callback=callback)


model.save("ppo_custom_env")


plt.figure(figsize=(12, 6))


if len(callback.episode_rewards) > 0:
    plt.subplot(1, 2, 1)
    plt.plot(callback.episode_rewards)
    plt.title("Raw Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    window_size = 20
    rewards_series = pd.Series(callback.episode_rewards)
    rolling_avg = rewards_series.rolling(window_size).mean()
    plt.plot(rolling_avg)
    plt.title(f"{window_size}-Episode Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.tight_layout()
    
    plt.savefig('training_results.png')
    plt.show()
else:
    print("No reward data to plot!")


obs = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()