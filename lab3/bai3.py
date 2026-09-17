import gymnasium as gym

env = gym.make("Blackjack-v1")

print("action_space:", env.action_space)
print("số lượng action:", env.action_space.n)

# Blackjack-v1 có action_space = Discrete(2), gồm 2 hành động:
#   0 = Stick : người chơi dừng lại, không rút thêm bài
#               (dealer sẽ bắt đầu rút bài theo luật của mình)
#   1 = Hit   : người chơi rút thêm 1 lá bài
#               (nếu tổng điểm > 21 thì bị "quắc" - bust và thua ngay)

ACTION_NAMES = {
    0: "Stick",  # Dừng, không rút thêm bài
    1: "Hit",    # Rút thêm 1 lá bài
}

for action, name in ACTION_NAMES.items():
    print(f"Action {action}: {name}")

env.close()