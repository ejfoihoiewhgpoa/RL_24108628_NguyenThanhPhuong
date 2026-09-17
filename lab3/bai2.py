import gymnasium as gym

env = gym.make("Blackjack-v1")

# Observation của Blackjack-v1 là 1 tuple gồm 3 phần tử:
# (player_sum, dealer_showing, usable_ace)
#
#   - player_sum      : tổng điểm các lá bài hiện có của người chơi (4 - 21)
#   - dealer_showing  : giá trị lá bài ngửa (duy nhất) của dealer (1 - 10)
#   - usable_ace      : True nếu người chơi có quân Át (Ace) tính là 11 điểm
#                        mà không bị "quắc" (bust); False nếu không có
#                        Át hoặc Át phải tính là 1 điểm

count = 0
episode = 0

while count < 10:
    observation, info = env.reset()
    episode += 1
    count += 1

    player_sum, dealer_showing, usable_ace = observation

    print(f"Episode {episode}:")
    print(f"observation = {observation}")
    print(f"player sum = {player_sum}")
    print(f"dealer showing = {dealer_showing}")
    print(f"usable ace = {usable_ace}")

env.close()