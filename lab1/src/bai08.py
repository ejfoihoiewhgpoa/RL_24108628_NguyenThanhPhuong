import gymnasium as gym

def run_one_step(env, action):
    return env.step(action)

env = gym.make("CartPole-v1")

observation, info = env.reset(seed = 42)

actions = []
for i in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = run_one_step(env, action)
    print(f"Step {i}: terminated: {terminated}, action: {action}, truncated: {truncated}, info: {info}")

env.close()