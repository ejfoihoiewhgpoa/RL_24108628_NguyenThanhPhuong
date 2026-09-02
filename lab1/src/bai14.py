import gymnasium as gym

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

print("Số episode:", len(episode_rewards))
print("Reward trung bình:", sum(episode_rewards) / len(episode_rewards))
print("Reward min/max:", min(episode_rewards), max(episode_rewards))