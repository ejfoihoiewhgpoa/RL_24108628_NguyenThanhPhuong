import gymnasium as gym

def run_experiment(is_slippery, total_episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

    success = 0
    failure = 0
    total_reward = 0
    total_length = 0

    for episode in range(total_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        episode_length = 0
        episode_reward = 0

        while not (terminated or truncated):
            action = env.action_space.sample()  # random policy
            state, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            episode_reward += reward

        if reward == 1.0:
            success += 1
        else:
            failure += 1

        total_reward += episode_reward
        total_length += episode_length

    env.close()

    success_rate = success / total_episodes
    average_reward = total_reward / total_episodes
    average_length = total_length / total_episodes

    return {
        "success": success,
        "failure": failure,
        "success_rate": success_rate,
        "average_reward": average_reward,
        "average_length": average_length,
    }

result_deterministic = run_experiment(is_slippery=False, total_episodes = 500)
result_stochastic = run_experiment(is_slippery=True, total_episodes = 500)

print("KẾT QUẢ: is_slippery = False (Deterministic)")
print(f"Success: {result_deterministic['success']}")
print(f"Failure: {result_deterministic['failure']}")
print(f"Success rate: {result_deterministic['success_rate']:.2%}")
print(f"Average reward: {result_deterministic['average_reward']:.4f}")
print(f"Average episode length: {result_deterministic['average_length']:.2f}")

print("\nKẾT QUẢ: is_slippery = True (Stochastic)")
print(f"Success: {result_stochastic['success']}")
print(f"Failure: {result_stochastic['failure']}")
print(f"Success rate: {result_stochastic['success_rate']:.2%}")
print(f"Average reward: {result_stochastic['average_reward']:.4f}")
print(f"Average episode length: {result_stochastic['average_length']:.2f}")

# ===================== KẾT LUẬN =====================
# 1) Success rate:
#    - Deterministic (is_slippery=False) có success rate rất thấp với random policy,
#      vì agent đi ngẫu nhiên không có định hướng, dễ rơi Hole trước khi tới Goal.
#    - Stochastic (is_slippery=True) success rate với random policy thường THẤP HƠN
#      hoặc xấp xỉ deterministic, vì thêm yếu tố trượt (agent có thể trượt sang
#      hướng khác dù chọn đúng action) khiến việc kiểm soát đường đi càng khó hơn.
#
# 2) Average reward:
#    - Vì reward trong FrozenLake chỉ là 0 hoặc 1 (nhận 1 khi tới Goal),
#      nên average reward = success rate. Trường hợp nào success rate thấp hơn
#      thì average reward cũng thấp hơn tương ứng.
#
# 3) Average episode length:
#    - Deterministic: episode length phụ thuộc hoàn toàn vào random walk,
#      thường ngắn hơn vì agent nhanh chóng rơi vào Hole hoặc đi hết vòng lặp.
#    - Stochastic: do agent có thể bị trượt (trượt ngang, đi ngược, lặp lại
#      vị trí cũ), episode có xu hướng KÉO DÀI HƠN trước khi kết thúc (rơi Hole,
#      tới Goal, hoặc chạm giới hạn bước truncation).
#
# => Kết luận chung: Stochastic environment (is_slippery=True) khiến hành vi
#    của agent trở nên khó dự đoán và khó kiểm soát hơn nhiều so với
#    Deterministic environment (is_slippery=False). Với random policy, cả 2
#    trường hợp đều có success rate thấp, nhưng để agent học tốt trong môi
#    trường stochastic, cần một policy được huấn luyện (không phải random)
#    để có thể tính đến rủi ro trượt ngẫu nhiên.