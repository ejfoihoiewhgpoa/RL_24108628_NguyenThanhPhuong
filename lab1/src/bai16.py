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
episode_lengths = []

for ep in range(100):
    total_reward, episode_length = random_agent(env)
    episode_rewards.append(total_reward)
    episode_lengths.append(episode_length)

env.close()

episode_rewards = np.array(episode_rewards)
episode_lengths = np.array(episode_lengths)

best_episode = np.argmax(episode_rewards)
best_reward = episode_rewards[best_episode]
best_length = episode_lengths[best_episode]

print(f"Best episode : {best_episode}")
print(f"Best reward  : {best_reward:.2f}")
print(f"Best length  : {best_length}")