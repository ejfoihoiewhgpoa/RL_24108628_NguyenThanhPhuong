import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

total_episodes = 100
success = 0
failure = 0

for episode in range(total_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample() 
        state, reward, terminated, truncated, info = env.step(action)

    if reward == 1.0:
        success += 1
    else:
        failure += 1

success_rate = success / total_episodes

print(f"Tổng số episode: {total_episodes}")
print(f"Success: {success}")
print(f"Failure: {failure}")
print(f"Success rate: {success_rate:.2%}")

env.close()