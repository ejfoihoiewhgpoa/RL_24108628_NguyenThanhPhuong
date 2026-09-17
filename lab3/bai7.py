def compute_returns(rewards, gamma=1.0):
    """
    Tính return G_t cho từng bước trong episode, đi ngược từ cuối lên đầu.

    Công thức:
        G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...
            = R_{t+1} + gamma * G_{t+1}

    Tham số:
        rewards : list các reward [R_1, R_2, ..., R_T]
                  (reward nhận được sau mỗi action, theo thứ tự thời gian)
        gamma   : hệ số discount (mặc định = 1.0, tức không discount,
                  phù hợp với Blackjack vì episode ngắn)

    Trả về:
        returns : list [G_0, G_1, ..., G_{T-1}]
                  tương ứng return tính từ mỗi bước cho đến hết episode
    """
    returns = [0] * len(rewards)
    G = 0

    # Duyệt ngược từ bước cuối cùng về bước đầu tiên
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


# ==== Ví dụ sử dụng ====
if __name__ == "__main__":
    import gymnasium as gym

    def generate_episode(env, policy, seed=None):
        episode = []
        state, info = env.reset(seed=seed) if seed is not None else env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    env = gym.make("Blackjack-v1")

    def random_policy(state):
        return env.action_space.sample()

    episode = generate_episode(env, random_policy, seed=42)
    rewards = [reward for (state, action, reward) in episode]

    returns = compute_returns(rewards, gamma=1.0)

    print("rewards:", rewards)
    print("returns:", returns)

    for t, (state, action, reward) in enumerate(episode):
        print(f"t={t}: state={state}, action={action}, reward={reward}, G_{t}={returns[t]}")

    env.close()