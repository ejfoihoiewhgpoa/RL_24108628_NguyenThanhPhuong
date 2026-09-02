import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

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

env = gym.make("CartPole-v1")

episode_rewards = []
episode_lengths = []

for ep in range(100):
    total_reward, episode_length = random_agent(env)
    episode_rewards.append(total_reward)
    episode_lengths.append(episode_length)

env.close()

os.makedirs("Lab01/figures", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
plt.title("Reward per Episode - Random Agent on CartPole-v1")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)

plt.savefig("lab1/figures/reward_cartpole.png")