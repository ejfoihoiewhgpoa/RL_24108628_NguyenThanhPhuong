import gymnasium as gym

env = gym.make("CartPole-v1")
env.reset(seed=42)

actions = []

for i in range(20):
    action = env.action_space.sample()
    actions.append(action)

print("Danh sach 20 action:", action)

count_0 = 0
count_1 = 0

for a in actions:
    if a == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1

print("So lan chon action 0:", count_0, "lan")
print("So lan chon action 1:", count_1, "lan")

env.close()