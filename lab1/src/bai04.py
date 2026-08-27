import gymnasium as gym

env = gym.make("CartPole-v1")
obs_space = env.observation_space

print("Shape:", obs_space.shape)
print("Data type:", obs_space.dtype)
print("Lower bound:", obs_space.low)
print("Upper bound:", obs_space.high)
