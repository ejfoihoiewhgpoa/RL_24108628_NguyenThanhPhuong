import gymnasium as gym

# ============================================================
# Ý nghĩa observation của CartPole (4 giá trị liên tục):
#   observation[0] = cart position       (vị trí xe)
#   observation[1] = cart velocity       (vận tốc xe)
#   observation[2] = pole angle          (góc nghiêng của pole, rad)
#   observation[3] = pole angular velocity (vận tốc góc của pole)
# ============================================================

env = gym.make("CartPole-v1")

def random_policy(observation):
    return env.action_space.sample()

def angle_based_policy(observation):
    pole_angle = observation[2]
    if pole_angle > 0:
        return 1   # pole nghiêng phải -> đẩy xe sang phải để đỡ lại
    else:
        return 0   # pole nghiêng trái -> đẩy xe sang trái để đỡ lại


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

avg_reward_random = run_policy(random_policy, total_episodes=100)
avg_reward_angle = run_policy(angle_based_policy, total_episodes=100)

env.close()

print("SO SÁNH KẾT QUẢ (100 episode mỗi policy)")
print(f"Random policy       -> Average reward: {avg_reward_random:.2f}")
print(f"Angle-based policy  -> Average reward: {avg_reward_angle:.2f}")

# ===================== KẾT LUẬN =====================
# Angle-based policy dựa trên pole_angle cho average reward CAO HƠN RÕ RỆT
# so với random policy, vì nó biết phản ứng ngược lại hướng nghiêng của pole
# để giữ thăng bằng, trong khi random policy chọn action hoàn toàn ngẫu nhiên
# không quan tâm đến trạng thái hiện tại.
#
# Tuy nhiên angle-based policy vẫn còn đơn giản (chỉ dùng 1/4 thông tin của
# observation - bỏ qua vận tốc góc, vị trí, vận tốc xe), nên vẫn chưa đạt được
# điểm tối đa (500) của CartPole. Đây là baseline cho thấy: chỉ cần dùng
# một phần nhỏ observation một cách hợp lý cũng đã cải thiện hiệu suất
# rất nhiều so với hành động ngẫu nhiên.
 