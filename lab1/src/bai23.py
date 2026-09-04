import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

print(env.observation_space)
print(env.action_space)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Số state:", n_states)
print("Số action:", n_actions)