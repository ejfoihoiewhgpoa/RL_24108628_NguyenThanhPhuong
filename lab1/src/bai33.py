import gymnasium as gym

def run_episode(env, policy, seed=None, max_steps=1000):
    """
    Chạy 1 episode tổng quát cho BẤT KỲ environment nào (không phụ thuộc
    riêng CartPole hay FrozenLake).
 
    Tham số:
        env       : environment đã được tạo bằng gym.make(...)
        policy    : hàm có dạng policy(observation) -> action
        seed      : seed cho env.reset() (tùy chọn, để tái lập kết quả)
        max_steps : số bước tối đa, phòng trường hợp env không tự truncate
 
    Trả về dictionary:
        {
            "reward": tổng reward cộng dồn cả episode,
            "length": số bước đã đi,
            "terminated": True/False,
            "truncated": True/False
        }
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

if __name__ == "__main__":

    def random_policy(observation):
        return env.action_space.sample()

    print("Thử với CartPole-v1")
    env = gym.make("CartPole-v1")
    result = run_episode(env, random_policy, seed=42)
    print(result)
    env.close()

    print("\nThử với FrozenLake-v1")
    env = gym.make("FrozenLake-v1", is_slippery=False)
    result = run_episode(env, random_policy, seed=42)
    print(result)
    env.close()