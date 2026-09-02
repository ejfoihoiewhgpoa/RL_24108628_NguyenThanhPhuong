import gymnasium as gym
import numpy as np

def random_agent(env, max_steps=500):
    observation, info = env.reset()
    total_reward = 0.0
    episode_length = 0

    for t in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_length += 1

        episode_finished = terminated or truncated
        if episode_finished:
            break

    return total_reward, episode_length


def run_episodes_with_seed(seed, num_episodes=20):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)          
    env.action_space.seed(seed)   

    rewards = []
    for ep in range(num_episodes):
        total_reward, episode_length = random_agent(env)
        rewards.append(total_reward)

    env.close()
    return rewards

rewards_42 = run_episodes_with_seed(seed=42)
rewards_100 = run_episodes_with_seed(seed=100)

mean_42 = np.mean(rewards_42)
mean_100 = np.mean(rewards_100)

print(f"Seed 42  - Mean reward: {mean_42:.2f}")
print(f"Seed 100 - Mean reward: {mean_100:.2f}")