import gymnasium as gym

env = gym.make("Blackjack-v1")

# ============================================================
# BƯỚC 1: Kiểm tra lại ý nghĩa action trước khi dùng
# ============================================================
# Theo tài liệu chính thức của Gymnasium (Blackjack-v1):
#   action_space = Discrete(2)
#   0 = Stick (dừng, không rút thêm bài)
#   1 = Hit   (rút thêm 1 lá bài)
#
# Ta có thể kiểm chứng nhanh bằng cách thử: nếu player_sum đã cao
# (ví dụ >= 20) mà chọn action, quan sát xem action nào khiến
# next_state không đổi (Stick không rút thêm bài -> state giữ nguyên
# phần player_sum) và action nào làm episode kết thúc ngay khi
# player_sum đang gần 21 (Hit dễ gây bust).

print("action_space:", env.action_space)

state, info = env.reset(seed=1)
print(f"\nThử nghiệm state = {state}")

# Thử action = 0
test_env = gym.make("Blackjack-v1")
s0, _ = test_env.reset(seed=1)
next_s0, r0, term0, trunc0, _ = test_env.step(0)
print(f"action=0 -> next_state={next_s0}, reward={r0}, terminated={term0}")
# Nếu terminated=True ngay và next_state[0] (player_sum) không đổi
# so với state ban đầu -> action 0 chính là Stick (dealer tự chơi
# và so bài ngay, không rút thêm bài của người chơi)

test_env.close()

# ============================================================
# BƯỚC 2: Định nghĩa policy cố định
# ============================================================
def stick_on_20_policy(state):
    """
    Policy đơn giản: nếu tổng điểm của người chơi >= 20 thì Stick,
    ngược lại thì Hit (rút thêm bài).
    """
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0  # Stick
    return 1      # Hit


# ============================================================
# BƯỚC 3: Thử policy này qua vài episode
# ============================================================
def generate_episode(env, policy, seed=None):
    episode = []
    state, info = env.reset(seed=seed) if seed is not None else env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


ACTION_NAMES = {0: "Stick", 1: "Hit"}

print("\n" + "=" * 60)
print("Thử stick_on_20_policy qua 5 episode:")
print("=" * 60)

for i in range(5):
    episode = generate_episode(env, stick_on_20_policy, seed=100 + i)
    print(f"\nEpisode {i + 1}:")
    for t, (state, action, reward) in enumerate(episode):
        print(f"  t={t}: state={state}, action={action} ({ACTION_NAMES[action]}), reward={reward}")
    total_reward = sum(r for (_, _, r) in episode)
    print(f"  -> Kết quả: {'THẮNG' if total_reward > 0 else 'THUA' if total_reward < 0 else 'HÒA'}")

env.close()