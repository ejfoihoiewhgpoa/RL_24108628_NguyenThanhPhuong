import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="ansi"
)

obs, info = env.reset()

print("Obs ban đầu:", obs)
print(env.render())