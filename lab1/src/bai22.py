import gymnasium as gym
import numpy as np


def experiment(seed, n_episodes):
    env = gym.make("CartPole-v1")

    env.reset(seed=seed)

    env.action_space.seed(seed)

    rewards = []

    for ep in range(n_episodes):
        observation, info = env.reset()

        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

        rewards.append(total_reward)

    env.close()

    return {
        "seed": seed,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards)
    }


def main():
    seeds = [0, 42, 100, 123, 999]

    n_episodes = 20

    for seed in seeds:
        result = experiment(seed, n_episodes)

        print(result)

if __name__ == "__main__":
    main()