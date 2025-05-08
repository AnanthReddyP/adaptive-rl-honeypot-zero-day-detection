import numpy as np
import gymnasium as gym
from typing import Optional, Dict

class HoneypotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2) 
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-0.5, 0.5, size=4)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        action_effects = {
            0: np.array([0.1, 0, 0, 0]),  
            1: np.array([-0.1, 0, 0, 0]), 
        }
        
        self.state = np.clip(self.state + action_effects[action], -1, 1)
        target = np.array([0.5, 0.5, 0, 0])
        distance = np.linalg.norm(self.state - target)
        reward = 10 * (1 - distance)  
        
        self.steps += 1
        done = distance < 0.1 or self.steps >= self.max_steps
        
        return self.state.copy(), reward, done, False, {}

    def render(self, mode='human'):
        print(f"\nState: {np.round(self.state, 2)}")
        print(f"Steps: {self.steps}")
        if hasattr(self, 'target'):
            print(f"Distance to target: {np.linalg.norm(self.state - self.target):.2f}")

    def test_rewards(self):
        print("\nReward Structure Test:")
        test_states = [
            np.array([0.5, 0.5, 0, 0]), 
            np.array([0, 0, 0, 0]),     
            np.array([1, 1, 0, 0])     
        ]
        
        for state in test_states:
            self.state = state.copy()
            for action in range(self.action_space.n):
                _, reward, _, _, _ = self.step(action)
                print(f"State {np.round(state, 2)} + Action {action} → Reward: {reward:.2f}")
            print()