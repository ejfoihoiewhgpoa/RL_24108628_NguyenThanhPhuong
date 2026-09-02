import gymnasium as gym
import numpy as np

initial_observations = []

for i in range(10):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    initial_observations.append(observation)
    env.close()

for i, obs in enumerate(initial_observations):
    print(f"Env {i}: {obs}")

all_same = all(np.allclose(initial_observations[0], obs) for obs in initial_observations)
print("\nTất cả observation giống nhau:", all_same)
