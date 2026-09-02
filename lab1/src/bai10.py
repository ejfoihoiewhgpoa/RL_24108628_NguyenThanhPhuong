import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

total_reward = 0.0

for t in range(20):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"t={t}, action={action}, reward={reward}")

    if terminated or truncated:
        print(f"Dừng ở timestep {t}")
        break

print(f"Episode length: {t + 1}")
print(f"Total reward: {total_reward}")

env.close()