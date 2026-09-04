import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def random_policy(observation, env):
    return env.action_space.sample()

def run_episode(env, policy, seed=None, max_steps=1000):
    observation, info = env.reset(seed = seed)

    if seed is not None:
        env.action_space.seed(seed)
    
    total_reward = 0

    for step in range(max_steps):
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward

def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    env = gym.make(env_name)

    rewards = []

    for i in range(n_episodes):
        episode_seed = seed + i

        reward = run_episode(
            env,
            policy,
            seed=episode_seed
        )

        rewards.append(reward)

    env.close()

    rewards = np.array(rewards)

    return rewards

def plot_rewards(rewards):
        plt.figure(figsize = (10, 5))
        plt.plot(range(1, len(rewards) + 1), rewards)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Random Policy Rewards")
        plt.grid(True)
        plt.show()

def main():
    env_name = "CartPole-v1"
    n_episodes = 100
    seed = 42

    rewards = evaluate_policy(
        env_name,
        random_policy,
        n_episodes=n_episodes,
        seed=seed
    )

    print("Environment:", env_name)
    print("Policy: Random Policy")
    print("Number of episodes:", n_episodes)
    print("Mean reward:", np.mean(rewards))
    print("Std reward:", np.std(rewards))
    print("Max reward:", np.max(rewards))
    print("Min reward:", np.min(rewards))

    plot_rewards(rewards)
if __name__ == "__main__":
    main()