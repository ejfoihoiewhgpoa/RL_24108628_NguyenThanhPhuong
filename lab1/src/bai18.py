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


def moving_average(values, window_size):
    """Tính moving average không dùng Pandas rolling."""
    values = np.array(values)
    result = np.zeros(len(values) - window_size + 1)
    for i in range(len(result)):
        result[i] = np.mean(values[i:i + window_size])
    return result

env = gym.make("CartPole-v1")

episode_rewards = []
episode_lengths = []

for ep in range(100):
    total_reward, episode_length = random_agent(env)
    episode_rewards.append(total_reward)
    episode_lengths.append(episode_length)

env.close()

window_size = 10
ma_rewards = moving_average(episode_rewards, window_size)

os.makedirs("Lab01/figures", exist_ok=True)

episodes = range(1, len(episode_rewards) + 1)
ma_episodes = range(window_size, len(episode_rewards) + 1)

plt.figure(figsize=(10, 6))
plt.plot(episodes, episode_rewards, label="Reward", alpha=0.4)
plt.plot(ma_episodes, ma_rewards, label=f"Moving Average (window={window_size})", linewidth=2)
plt.title("Reward per Episode with Moving Average - Random Agent on CartPole-v1")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()

plt.savefig("lab1/figures/moving_average.png")