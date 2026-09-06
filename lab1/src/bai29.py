import gymnasium as gym

def policy(observation):
    """
    Policy ban đầu: chỉ trả về action ngẫu nhiên,
    chưa dùng thông tin gì từ observation.
    """
    return env.action_space.sample()


env = gym.make("FrozenLake-v1", is_slippery=False)

total_episodes = 500
success = 0
failure = 0
total_reward = 0
total_length = 0

for episode in range(total_episodes):
    observation, info = env.reset()
    terminated = False
    truncated = False
    episode_length = 0
    episode_reward = 0

    while not (terminated or truncated):
        action = policy(observation) 
        observation, reward, terminated, truncated, info = env.step(action)
        episode_length += 1
        episode_reward += reward

    if reward == 1.0:
        success += 1
    else:
        failure += 1

    total_reward += episode_reward
    total_length += episode_length

env.close()

success_rate = success / total_episodes
average_reward = total_reward / total_episodes
average_length = total_length / total_episodes

print(f"Success: {success}")
print(f"Failure: {failure}")
print(f"Success rate: {success_rate:.2%}")
print(f"Average reward: {average_reward:.4f}")
print(f"Average episode length: {average_length:.2f}")