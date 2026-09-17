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


def compute_returns(rewards, gamma=1.0):
    returns = [0] * len(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def attach_returns(episode, gamma=1.0):
    """
    Gắn return G_t vào từng bước của episode.

    Tham số:
        episode : list các tuple (state, action, reward)
        gamma   : hệ số discount

    Trả về:
        trajectory : list các tuple (state, action, reward, G_t)
    """
    rewards = [reward for (state, action, reward) in episode]
    returns = compute_returns(rewards, gamma=gamma)

    trajectory = [
        (state, action, reward, G)
        for (state, action, reward), G in zip(episode, returns)
    ]
    return trajectory


# ==== Ví dụ sử dụng ====
if __name__ == "__main__":
    env = gym.make("Blackjack-v1")

    def random_policy(state):
        return env.action_space.sample()

    episode = generate_episode(env, random_policy, seed=42)
    trajectory = attach_returns(episode, gamma=1.0)

    print("Toàn bộ trajectory (state, action, reward, G_t):")
    for t, (state, action, reward, G) in enumerate(trajectory):
        print(f"t={t}: state={state}, action={action}, reward={reward}, G_{t}={G}")

    env.close()