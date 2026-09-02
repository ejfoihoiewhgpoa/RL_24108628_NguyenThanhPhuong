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

env = gym.make("CartPole-v1")

episode_rewards = []
for ep in range(100):
    total_reward, episode_length = random_agent(env)
    episode_rewards.append(total_reward)

env.close()

episode_rewards = np.array(episode_rewards)

mean_reward = np.mean(episode_rewards)
min_reward = np.min(episode_rewards)
max_reward = np.max(episode_rewards)
std_reward = np.std(episode_rewards)

print(f"Mean reward : {mean_reward:.2f}")
print(f"Min reward  : {min_reward:.2f}")
print(f"Max reward  : {max_reward:.2f}")
print(f"Std reward  : {std_reward:.2f}")