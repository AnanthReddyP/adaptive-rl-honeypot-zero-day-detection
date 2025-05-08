import numpy as np
import torch
from honeypot_env import HoneypotEnv
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def run_evaluation(episodes=10, render=False):
    env = HoneypotEnv()
    try:
        model = DQN.load("honeypot_rl_agent")
    except FileNotFoundError:
        raise FileNotFoundError("Model file 'honeypot_rl_agent.zip' not found")
    

    episode_rewards = []
    action_counts = defaultdict(int)
    state_history = []
    action_history = []
    reward_history = []
    q_values = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 100:
            
            if isinstance(state, tuple):
                state = state[0]
            state = np.array(state, dtype=np.float32)
            
            with torch.no_grad():
                state_tensor = torch.as_tensor(state).float().unsqueeze(0) 
                current_q = model.policy.q_net(state_tensor).squeeze(0).numpy() 
                q_values.append(current_q)
            
            action, _ = model.predict(state, deterministic=True)
            action = int(action) 
            
            next_state, reward, done, _, _ = env.step(action)
            
            
            action_counts[action] += 1
            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)
            total_reward += reward
            
            if render:
                env.render()
                print(f"Episode {ep+1}, Step {step}:")
                print(f"State: {np.round(state, 2)}")
                print(f"Action: {action}")
                print(f"Reward: {reward:.2f}")
                print(f"Q-values: {np.round(current_q, 2)}")
                print(f"Next State: {np.round(next_state, 2)}")
                print("-"*40)
            
            state = next_state
            step += 1
        
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1} finished with total reward: {total_reward:.2f}")


    print("\n=== Evaluation Metrics ===")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Action Distribution: {dict(action_counts)}")
    print(f"State Mean: {np.mean(state_history, axis=0)}")
    print(f"Reward Range: {np.min(reward_history):.2f} to {np.max(reward_history):.2f}")

    
    plt.figure(figsize=(15, 5))

   
    plt.subplot(1, 3, 1)
    q_values = np.array(q_values)
    for i in range(q_values.shape[1]):
        plt.plot(q_values[:100, i], label=f'Action {i}', alpha=0.7)
    plt.title("Q-value Progression")
    plt.xlabel("Step")
    plt.ylabel("Q-value")
    plt.legend()

  
    plt.subplot(1, 3, 2)
    plt.bar(action_counts.keys(), action_counts.values())
    plt.title("Action Frequency")
    plt.xlabel("Action")
    plt.ylabel("Count")

    
    plt.subplot(1, 3, 3)
    window_size = 5
    rewards_series = pd.Series(episode_rewards)
    rolling_avg = rewards_series.rolling(window_size).mean()
    plt.plot(episode_rewards, 'o-', alpha=0.3, label='Raw')
    plt.plot(rolling_avg, 'r-', label=f'{window_size}-ep Avg')
    plt.axhline(np.mean(episode_rewards), color='g', linestyle='--', label='Mean')
    plt.title("Reward Progression")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.tight_layout()
    plt.savefig('detailed_evaluation.png', dpi=300)
    plt.show()

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'action_distribution': dict(action_counts),
        'q_values': q_values
    }

if __name__ == "__main__":
    
    print("Testing environment rewards...")
    env = HoneypotEnv()
    env.test_rewards()
    
    
    print("\nRunning agent evaluation...")
    results = run_evaluation(episodes=10, render=True)
    
    print("\n=== Final Evaluation Summary ===")
    print(f"Average Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Unique Actions Taken: {len(results['action_distribution'])}")
    print(f"Q-value Range: {np.min(results['q_values']):.2f} to {np.max(results['q_values']):.2f}")