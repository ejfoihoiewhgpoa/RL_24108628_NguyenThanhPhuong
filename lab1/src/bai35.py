import os
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_episode(env, policy, seed=None, max_steps=1000):
    if seed is not None:
        observation, info = env.reset(seed=seed)
    else:
        observation, info = env.reset()

    total_reward = 0
    length = 0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
    }


def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    env = gym.make(env_name)
    rewards, lengths = [], []

    for i in range(n_episodes):
        result = run_episode(env, policy, seed=seed + i)
        rewards.append(result["reward"])
        lengths.append(result["length"])

    env.close()
    rewards = np.array(rewards)
    lengths = np.array(lengths)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


# ===================== 3 POLICY =====================

def random_policy(observation):
    return _tmp_env.action_space.sample()

_tmp_env = gym.make("CartPole-v1")  # dùng để random_policy biết action_space


def angle_based_policy(observation):
    """Bài 31: chỉ dùng pole_angle."""
    pole_angle = observation[2]
    return 1 if pole_angle > 0 else 0


def improved_policy(observation):
    """
    Bài 32: kết hợp pole_angle và pole_angular_velocity để phản ứng
    đón đầu xu hướng nghiêng của pole.
    """
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    combined = pole_angle + 0.5 * pole_angular_velocity
    return 1 if combined > 0 else 0


# ===================== CHẠY ĐÁNH GIÁ 3 AGENT (500 episode) =====================

N_EPISODES = 500

agents = {
    "Random": random_policy,
    "Angle-based": angle_based_policy,
    "Improved": improved_policy,
}

results = {}
for name, policy in agents.items():
    results[name] = evaluate_policy("CartPole-v1", policy, n_episodes=N_EPISODES, seed=42)

_tmp_env.close()

# ===================== IN BẢNG KẾT QUẢ =====================

print(f"{'Agent':<15}{'Mean reward':>13}{'Std':>10}{'Min':>8}{'Max':>8}{'Mean length':>14}")
for name, r in results.items():
    print(f"{name:<15}{r['mean_reward']:>13.2f}{r['std_reward']:>10.2f}"
          f"{r['min_reward']:>8.2f}{r['max_reward']:>8.2f}{r['mean_length']:>14.2f}")

# ===================== VẼ BIỂU ĐỒ =====================

output_dir = "lab1/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "comparison_agents.png")

names = list(results.keys())
means = [results[n]["mean_reward"] for n in names]
stds = [results[n]["std_reward"] for n in names]

plt.figure(figsize=(7, 5))
bars = plt.bar(names, means, yerr=stds, capsize=8, color=["#888888", "#4C72B0", "#55A868"])
plt.ylabel("Mean reward (500 episodes)")
plt.title("So sánh Mean Reward của 3 Agent trên CartPole-v1")

for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
              f"{mean:.1f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(output_path, dpi=150)
with open(os.path.join(output_dir, "comparison_agents_table.txt"), "w", encoding="utf-8") as f:
    for name, r in results.items():
        f.write(f"{name:<15}{r['mean_reward']:>13.2f}{r['std_reward']:>10.2f}{r['min_reward']:>8.2f}{r['max_reward']:>8.2f}{r['mean_length']:>14.2f}\n")
plt.close()

print(f"\nĐã lưu biểu đồ tại: {output_path}")

fig, ax = plt.subplots(figsize=(7, 2))
ax.axis("off")
 
table_data = [["Agent", "Mean reward", "Std", "Min", "Max", "Mean length"]]
for name, r in results.items():
    table_data.append([
        name,
        f"{r['mean_reward']:.2f}",
        f"{r['std_reward']:.2f}",
        f"{r['min_reward']:.2f}",
        f"{r['max_reward']:.2f}",
        f"{r['mean_length']:.2f}",
    ])
 
table = ax.table(cellText=table_data, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
 
# Tô đậm hàng tiêu đề
for col in range(len(table_data[0])):
    table[(0, col)].set_facecolor("#4C72B0")
    table[(0, col)].set_text_props(color="white", fontweight="bold")
 
plt.tight_layout()
table_image_path = os.path.join(output_dir, "comparison_agents_table.png")
plt.savefig(table_image_path, dpi=150, bbox_inches="tight")
plt.close()
 
print(f"Đã lưu bảng kết quả (ảnh) tại: {table_image_path}")

# ===================== NHẬN XÉT (5-10 dòng) =====================
# 1. Random policy có mean reward thấp nhất và std tương đối nhỏ, vì hành động
#    ngẫu nhiên hầu như luôn khiến pole đổ rất nhanh và khá đều đặn ở mức thấp.
# 2. Angle-based policy (chỉ dùng pole_angle) cải thiện đáng kể so với random,
#    vì nó biết phản ứng ngược hướng nghiêng, nhưng vẫn phản ứng "trễ" một nhịp
#    do chỉ nhìn góc hiện tại mà không biết pole đang nghiêng nhanh hay chậm.
# 3. Improved policy (kết hợp pole_angle và pole_angular_velocity) đạt mean
#    reward cao vượt trội, nhiều episode chạm mức tối đa 500, vì nó "đón đầu"
#    được xu hướng nghiêng thay vì chỉ phản ứng khi pole đã lệch nhiều.
# 4. Std của Improved policy có thể lớn hơn Angle-based ở một số trường hợp vì
#    khi đạt điểm cao (gần 500), chỉ cần một vài episode kết thúc sớm do rơi
#    vào vùng trạng thái xấu cũng đủ kéo phương sai lên, trong khi các policy
#    yếu hơn luôn kết thúc sớm nên std tự nhiên nhỏ và ổn định "ở đáy".
# 5. Kết quả cho thấy: chỉ cần thêm MỘT thành phần observation phù hợp
#    (pole_angular_velocity) đã tạo ra khác biệt rất lớn về hiệu suất, minh
#    chứng tầm quan trọng của việc lựa chọn đặc trưng (feature) phù hợp
#    khi thiết kế policy, ngay cả với một heuristic đơn giản (if-else),
#    trước khi cần đến các thuật toán học tăng cường phức tạp hơn.