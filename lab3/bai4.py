import gymnasium as gym

env = gym.make("Blackjack-v1")

ACTION_NAMES = {
    0: "Stick",
    1: "Hit",
}

state, info = env.reset()
terminated = False
truncated = False
step = 0

print("Bắt đầu episode mới")
print(f"State ban đầu: {state}")
print("=" * 60)

while not terminated and not truncated:
    step += 1

    # Random policy: chọn ngẫu nhiên action trong action_space
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step}:")
    print(f"state = {state}")
    print(f"action = {action} ({ACTION_NAMES[action]})")
    print(f"reward = {reward}")
    print(f"next_state = {next_state}")
    print(f"terminated = {terminated}")
    print(f"truncated = {truncated}")
    
    state = next_state

print("Episode kết thúc")
if reward > 0:
    print("Kết quả: Người chơi THẮNG")
elif reward < 0:
    print("Kết quả: Người chơi THUA")
else:
    print("Kết quả: HÒA")

env.close()