import gymnasium as gym


def generate_episode(env, policy, seed=None):
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
        episode.append((state, action, reward))
        state = next_state

    return episode


env = gym.make("Blackjack-v1")

def random_policy(state):
    return env.action_space.sample()

# Tạo episode
episode = generate_episode(env, random_policy, seed=42)

# Lấy chuỗi reward từ episode
# Mỗi phần tử episode[i] = (state, action, reward)
rewards = [reward for (state, action, reward) in episode]

# In kết quả
print("episode length:", len(episode))
print("rewards:", rewards)
print("total reward:", sum(rewards))

env.close()