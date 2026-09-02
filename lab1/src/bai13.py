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

print(f"{'Episode':<10}{'Reward':<10}{'Length':<10}")
for ep in range(1, 11):
    total_reward, episode_length = random_agent(env)
    print(f"{ep:<10}{total_reward:<10}{episode_length:<10}")

env.close()