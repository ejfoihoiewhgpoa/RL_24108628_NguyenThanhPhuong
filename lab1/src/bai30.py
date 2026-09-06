import gymnasium as gym

def always_left_policy(observation):
    return 0  # CartPole: action 0 = đẩy xe sang trái

def always_right_policy(observation):
    return 1  # CartPole: action 1 = đẩy xe sang phải


def run_policy(policy, total_episodes=100):
    env = gym.make("CartPole-v1")
    total_reward = 0
    total_length = 0

    for episode in range(total_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (terminated or truncated):
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        total_reward += episode_reward
        total_length += episode_length

    env.close()

    average_reward = total_reward / total_episodes
    average_length = total_length / total_episodes
    return average_reward, average_length


avg_reward_left, avg_length_left = run_policy(always_left_policy, total_episodes=100)
avg_reward_right, avg_length_right = run_policy(always_right_policy, total_episodes=100)

print("always_left_policy")
print(f"Average reward: {avg_reward_left:.2f}")
print(f"Average episode length: {avg_length_left:.2f}")

print("\nalways_right_policy ")
print(f"Average reward: {avg_reward_right:.2f}")
print(f"Average episode length: {avg_length_right:.2f}")

 
# ===================== KẾT LUẬN =====================
# Trong CartPole, mỗi bước sống sót (chưa bị ngã) được +1 reward, nên
# average reward chính là số bước trung bình mà cây gậy đứng được trước khi đổ.
#
# Cả always_left_policy và always_right_policy đều là các policy "cứng nhắc"
# (không quan tâm đến observation - tức góc nghiêng, vận tốc của xe/gậy),
# nên cây gậy sẽ nhanh chóng ngã về một phía do lực đẩy liên tục theo 1 hướng.
# => Cả hai đều cho reward trung bình THẤP và gần bằng nhau (thường dưới ~10),
#    vì bản chất 2 policy này đối xứng nhau, chỉ khác chiều đẩy.
#
# So với random policy (chọn ngẫu nhiên 0/1), 2 policy cố định này thường
# cho kết quả KÉM HƠN, vì random policy đôi khi vô tình đẩy đúng chiều
# để cân bằng lại gậy, còn policy cố định thì luôn đẩy 1 chiều nên gậy đổ rất nhanh.