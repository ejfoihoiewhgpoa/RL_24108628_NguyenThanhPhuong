import gymnasium as gym


def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0      # Stick
    return 1          # Hit

env = gym.make("Blackjack-v1")

# Blackjack-v1:
# action = 0 -> Stick
# action = 1 -> Hit

num_episodes = 100

win = 0
loss = 0
draw = 0

for i in range(num_episodes):

    state, info = env.reset(seed=1000 + i)

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Policy chọn action
        action = stick_on_20_policy(state)

        # Thực hiện action
        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state

    if reward > 0:
        win += 1
    elif reward < 0:
        loss += 1
    else:
        draw += 1


env.close()


win_rate = win / num_episodes * 100
loss_rate = loss / num_episodes * 100
draw_rate = draw / num_episodes * 100

print("KẾT QUẢ 100 EPISODE")
print(f"Win  : {win}  ({win_rate:.2f}%)")
print(f"Loss : {loss}  ({loss_rate:.2f}%)")
print(f"Draw : {draw}  ({draw_rate:.2f}%)")