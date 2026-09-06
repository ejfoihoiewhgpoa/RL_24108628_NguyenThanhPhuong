import gymnasium as gym
import numpy as np


def run_episode(env, policy, seed=None, max_steps=1000):
    """
    Hàm chạy 1 episode tổng quát (đã xây dựng ở Bài 33).
    Trả về dict: {"reward", "length", "terminated", "truncated"}
    """
    if seed is not None:
        observation, info = env.reset(seed=seed)
    else:
        observation, info = env.reset()

    total_reward = 0
    length = 0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        length += 1

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
    }


def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    """
    Đánh giá 1 policy trên bất kỳ environment nào (dựa theo tên env_name),
    chạy n_episodes lần, mỗi lần dùng seed khác nhau (seed + i) để đảm bảo
    kết quả đa dạng nhưng vẫn có thể tái lập (reproducible).
 
    Trả về dictionary:
        {
            "mean_reward": trung bình reward,
            "std_reward": độ lệch chuẩn reward,
            "min_reward": reward thấp nhất,
            "max_reward": reward cao nhất,
            "mean_length": trung bình số bước mỗi episode
        }
    """
    env = gym.make(env_name)

    rewards = []
    lengths = []

    for i in range(n_episodes):
        episode_seed = seed + i  # mỗi episode 1 seed khác nhau nhưng tái lập được
        result = run_episode(env, policy, seed=episode_seed)
        rewards.append(result["reward"])
        lengths.append(result["length"])

    env.close()

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
    }

if __name__ == "__main__":

    def make_random_policy(action_space):
        def policy(observation):
            return action_space.sample()
        return policy

    print("Đánh giá random_policy trên CartPole-v1")
    tmp_env = gym.make("CartPole-v1")
    random_policy_cartpole = make_random_policy(tmp_env.action_space)
    tmp_env.close()

    result = evaluate_policy("CartPole-v1", random_policy_cartpole, n_episodes=100, seed=42)
    for k, v in result.items():
        print(f"{k}: {v:.2f}")

    print("\nĐánh giá random_policy trên FrozenLake-v1")
    tmp_env = gym.make("FrozenLake-v1", is_slippery=False)
    random_policy_frozenlake = make_random_policy(tmp_env.action_space)
    tmp_env.close()

    result = evaluate_policy("FrozenLake-v1", random_policy_frozenlake, n_episodes=100, seed=42)
    for k, v in result.items():
        print(f"{k}: {v:.2f}")