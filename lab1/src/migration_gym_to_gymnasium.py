import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")  # đổi v0 -> v1

observation, info = env.reset(seed=42)

for t in range(1000):
    env.render()

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    # Với CartPole-v1, nếu chạy đủ 500 bước mà cây gậy chưa đổ:
    # -> truncated = True (do đạt max_steps=500), terminated = False
    # Nếu cây gậy đổ trước đó:
    # -> terminated = True

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()