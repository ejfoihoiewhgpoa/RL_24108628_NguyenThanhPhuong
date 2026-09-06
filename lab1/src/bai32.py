import gymnasium as gym

# ============================================================
# Ý nghĩa observation của CartPole:
#   observation[0] = cart position
#   observation[1] = cart velocity
#   observation[2] = pole angle              (góc nghiêng)
#   observation[3] = pole angular velocity   (tốc độ nghiêng)
# ============================================================

env = gym.make("CartPole-v1")

def random_policy(observation):
    return env.action_space.sample()


def angle_based_policy(observation):
    pole_angle = observation[2]
    return 1 if pole_angle > 0 else 0


def improved_policy(observation):
    """
    Policy bài 32 - dùng CẢ pole_angle VÀ pole_angular_velocity.
 
    Ý tưởng: không chỉ nhìn góc nghiêng hiện tại, mà còn nhìn tốc độ
    nghiêng (pole đang nghiêng thêm nhanh hay chậm) để phản ứng SỚM hơn,
    tránh việc luôn phản ứng trễ 1 nhịp như policy chỉ dùng góc.
 
    Công thức: combined = pole_angle + 0.5 * pole_angular_velocity
      - Nếu combined > 0 (pole đang/sắp nghiêng phải) -> đẩy xe sang phải
      - Nếu combined <= 0 (pole đang/sắp nghiêng trái) -> đẩy xe sang trái
    """
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]

    combined = pole_angle + 0.5 * pole_angular_velocity

    if combined > 0:
        return 1  # RIGHT
    else:
        return 0  # LEFT


def run_policy(policy, total_episodes=100):
    total_reward = 0
    for episode in range(total_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / total_episodes


env = gym.make("CartPole-v1")

avg_random = run_policy(random_policy, total_episodes=100)
avg_angle_only = run_policy(angle_based_policy, total_episodes=100)
avg_improved = run_policy(improved_policy, total_episodes=100)

env.close()

print("SO SÁNH KẾT QUẢ (100 episode mỗi policy)")
print(f"Random policy              -> Average reward: {avg_random:.2f}")
print(f"Angle-based policy (Bài 31) -> Average reward: {avg_angle_only:.2f}")
print(f"Improved policy (Bài 32)   -> Average reward: {avg_improved:.2f}")

# ===================== KẾT LUẬN =====================
# Improved policy (dùng cả pole_angle và pole_angular_velocity) cho
# average reward CAO HƠN NHIỀU so với cả random policy lẫn angle-based
# policy chỉ dùng 1 biến (Bài 31), thậm chí đạt gần mức tối đa (500).
#
# Lý do: pole_angle chỉ cho biết pole đang nghiêng bao nhiêu TẠI THỜI ĐIỂM
# HIỆN TẠI, còn pole_angular_velocity cho biết pole đang nghiêng NHANH hay
# CHẬM, tức xu hướng trong tương lai gần. Kết hợp cả 2 giúp agent phản ứng
# "đón đầu" (dự đoán trước) thay vì chỉ phản ứng khi pole đã nghiêng hẳn,
# nhờ đó giữ thăng bằng ổn định và lâu hơn nhiều.
#
# => Mục tiêu đạt được: mean reward của improved_policy > mean reward
#    của random_policy (và cũng > angle-based policy của Bài 31).