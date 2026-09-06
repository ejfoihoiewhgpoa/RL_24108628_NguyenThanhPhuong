import gymnasium as gym

env = gym.make("FrozenLake-v1")

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

action = env.action_space.sample()
print(f"Action {action} -> {ACTION_NAMES[action]}")