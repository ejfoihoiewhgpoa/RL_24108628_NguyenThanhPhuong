import gymnasium as gym 

env = gym.make("CartPole-v1")

observation, info = env.reset(seed = 42)

actions = []
for i in range(1):
    action = env.action_space.sample()
    actions.append(action)

observation, reward, terminated, truncated, info = env.step(action)

print("Action:", action)
print("State after action:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info", info)

env.close()