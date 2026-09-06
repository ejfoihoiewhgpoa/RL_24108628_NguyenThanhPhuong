import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

actions = [2, 2, 1, 1, 1, 2]

print(f"Trạng thái ban đầu: {state}")

for step, action in enumerate(actions, start=1):
    state, reward, terminated, truncated, info = env.step(action)
    print(f"Bước {step}: Action {action} ({ACTION_NAMES[action]}) -> State {state}, "
          f"Reward={reward}, Terminated={terminated}")

    if terminated:
        if reward == 1.0:
            print(">>> Đã tới Goal thành công!")
        else:
            print(">>> Rơi vào Hole, thất bại!")
        break

env.close()