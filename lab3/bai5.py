import gymnasium as gym


def generate_episode(env, policy, seed=None):
    """
    Chạy 1 episode theo policy cho trước và lưu lại toàn bộ lịch sử.

    Tham số:
        env    : môi trường gymnasium (đã gym.make(...))
        policy : hàm nhận vào state, trả về action
                 (policy(state) -> action)
        seed   : (tùy chọn) giá trị seed để reset môi trường,
                 giúp tái lập lại kết quả khi cần

    Trả về:
        episode : list các tuple (state, action, reward)
                  ghi lại từng bước đã đi qua trong episode
    """
    episode = []

    if seed is not None:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Lưu lại (state, action, reward) của bước này
        episode.append((state, action, reward))

        state = next_state

    return episode


# ==== Ví dụ sử dụng ====
if __name__ == "__main__":
    env = gym.make("Blackjack-v1")

    # Random policy: chọn ngẫu nhiên action
    def random_policy(state):
        return env.action_space.sample()

    episode = generate_episode(env, random_policy, seed=42)

    print("Episode:")
    for i, (state, action, reward) in enumerate(episode, 1):
        print(f"Step {i}: state={state}, action={action}, reward={reward}")

    print(f"Tổng số bước: {len(episode)}")
    print(f"Tổng reward: {sum(r for _, _, r in episode)}")

    env.close()