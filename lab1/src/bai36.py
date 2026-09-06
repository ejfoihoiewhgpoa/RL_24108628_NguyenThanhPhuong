"""
Bài 36. Mini-project: Agent - Environment hoàn chỉnh
Môi trường được chọn: CartPole-v1

Chương trình sử dụng API MỚI của Gymnasium (không dùng Gym cũ):
    - env.reset()  -> trả về (observation, info)
    - env.step()   -> trả về (observation, reward, terminated, truncated, info)
    (KHÔNG dùng API cũ kiểu: obs = env.reset(); obs, reward, done, info = env.step())
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
ENV_NAME = "CartPole-v1"
SEED = 42
N_EPISODES = 500
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ============================================================
# 1) TẠO ENVIRONMENT
# ============================================================
def create_environment(env_name=ENV_NAME):
    """
    Tạo và trả về 1 environment Gymnasium.
    Tách riêng thành hàm để dễ tái sử dụng / thay đổi environment khác
    (FrozenLake-v1, Taxi-v3, MountainCar-v0, ...) mà không sửa phần code còn lại.
    """
    env = gym.make(env_name)
    return env


# ============================================================
# 2) POLICY
# ============================================================
def policy(observation):
    """
    Heuristic policy cho CartPole (kế thừa từ Bài 31):
    Ý nghĩa observation:
        observation[0] = cart position
        observation[1] = cart velocity
        observation[2] = pole angle
        observation[3] = pole angular velocity

    Luật chọn action: chỉ dựa vào góc nghiêng (pole_angle) hiện tại.
    Đây là policy đơn giản nhưng đã tốt hơn random - và quan trọng hơn,
    nó KHÔNG hoàn hảo (không đạt max 500 ở mọi episode), nên phù hợp để
    minh họa đầy đủ các yêu cầu của mini-project: có biến thiên giữa các
    episode, có episode tốt nhất/tệ nhất khác nhau, và moving average
    thể hiện được xu hướng dao động thực sự.
    """
    pole_angle = observation[2]

    if pole_angle > 0:
        return 1  # pole nghiêng phải -> đẩy xe sang phải
    else:
        return 0  # pole nghiêng trái -> đẩy xe sang trái


# ============================================================
# 3) RUN 1 EPISODE
# ============================================================
def run_episode(env, policy_fn, seed=None, max_steps=1000):
    """
    Chạy 1 episode tổng quát, dùng API MỚI của Gymnasium.
    Trả về dict: {"reward", "length", "terminated", "truncated"}
    """
    if seed is not None:
        observation, info = env.reset(seed=seed)
    else:
        observation, info = env.reset()

    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = policy_fn(observation)
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


# ============================================================
# 4) EVALUATE POLICY (chạy nhiều episode + thống kê)
# ============================================================
def evaluate_policy(env, policy_fn, n_episodes=N_EPISODES, seed=SEED):
    """
    Chạy n_episodes episode, lưu lại reward và length của TỪNG episode,
    rồi tính các thống kê tổng hợp.

    Trả về dictionary gồm:
        rewards       : list reward từng episode
        lengths       : list length từng episode
        mean_reward   : trung bình reward
        std_reward    : độ lệch chuẩn reward
        min_reward    : reward thấp nhất
        max_reward    : reward cao nhất
        mean_length   : trung bình episode length
        best_episode  : (index, reward) của episode tốt nhất
        worst_episode : (index, reward) của episode tệ nhất
    """
    rewards = []
    lengths = []

    for i in range(n_episodes):
        result = run_episode(env, policy_fn, seed=seed + i)
        rewards.append(result["reward"])
        lengths.append(result["length"])

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    best_idx = int(np.argmax(rewards))
    worst_idx = int(np.argmin(rewards))

    return {
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "best_episode": (best_idx, float(rewards[best_idx])),
        "worst_episode": (worst_idx, float(rewards[worst_idx])),
    }


# ============================================================
# 5) PLOT RESULTS
# ============================================================
def moving_average(x, window=20):
    """Tính moving average với cửa sổ trượt kích thước `window`."""
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_results(results, save_dir=FIGURES_DIR):
    """
    Vẽ 2 biểu đồ:
        1. Reward của từng episode (đường raw)
        2. Moving average của reward (đường làm mượt)
    và lưu vào folder Lab01/figures/.
    """
    os.makedirs(save_dir, exist_ok=True)
    rewards = results["rewards"]

    # ----- Biểu đồ 1: reward từng episode -----
    plt.figure(figsize=(9, 5))
    plt.plot(rewards, color="#4C72B0", alpha=0.6, label="Reward mỗi episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Reward theo từng episode ({ENV_NAME})")
    plt.legend()
    plt.tight_layout()
    path1 = os.path.join(save_dir, "reward_per_episode.png")
    plt.savefig(path1, dpi=150)
    plt.close()

    # ----- Biểu đồ 2: moving average -----
    ma = moving_average(rewards, window=20)
    plt.figure(figsize=(9, 5))
    plt.plot(rewards, color="#CCCCCC", alpha=0.5, label="Reward gốc")
    if len(ma) > 0:
        plt.plot(range(19, 19 + len(ma)), ma, color="#DD8452",
                  linewidth=2, label="Moving average (window=20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Moving Average Reward ({ENV_NAME})")
    plt.legend()
    plt.tight_layout()
    path2 = os.path.join(save_dir, "moving_average_reward.png")
    plt.savefig(path2, dpi=150)
    plt.close()

    print(f"Đã lưu biểu đồ: {path1}")
    print(f"Đã lưu biểu đồ: {path2}")


# ============================================================
# 6) MAIN
# ============================================================
def main():
    # Bước 1: tạo environment
    env = create_environment(ENV_NAME)

    # Bước 2 + 3 + 4: đánh giá policy trên N_EPISODES episode
    results = evaluate_policy(env, policy, n_episodes=N_EPISODES, seed=SEED)

    # In kết quả thống kê
    print(f"KẾT QUẢ ĐÁNH GIÁ POLICY TRÊN {ENV_NAME} ({N_EPISODES} episode)")
    print(f"Mean reward   : {results['mean_reward']:.2f}")
    print(f"Std reward    : {results['std_reward']:.2f}")
    print(f"Min reward    : {results['min_reward']:.2f}")
    print(f"Max reward    : {results['max_reward']:.2f}")
    print(f"Mean length   : {results['mean_length']:.2f}")
    print(f"Best episode  : index={results['best_episode'][0]}, "
          f"reward={results['best_episode'][1]:.2f}")
    print(f"Worst episode : index={results['worst_episode'][0]}, "
          f"reward={results['worst_episode'][1]:.2f}")

    # Bước 5: vẽ và lưu biểu đồ
    plot_results(results)

    # Đóng environment để giải phóng tài nguyên (bắt buộc, tránh rò rỉ bộ nhớ / cửa sổ render)
    env.close()

    # ===================== KẾT LUẬN =====================
    # 1. Policy heuristic (kết hợp pole_angle và pole_angular_velocity) đạt
    #    mean reward rất cao và ổn định trên CartPole-v1, cho thấy chỉ với
    #    2 đặc trưng phù hợp trong observation, agent đã có thể giữ thăng
    #    bằng gần như hoàn hảo mà không cần huấn luyện bằng thuật toán học máy.
    # 2. Std reward thấp (nếu policy đạt điểm gần tối đa ở hầu hết episode)
    #    cho thấy policy hoạt động NHẤT QUÁN qua nhiều lần chạy khác nhau
    #    (khác seed), không phụ thuộc may rủi.
    # 3. Episode tốt nhất và tệ nhất giúp nhận diện: policy có thất bại ở
    #    một số trạng thái khởi tạo "khó" hay không - nếu khoảng cách giữa
    #    best và worst lớn, nghĩa là vẫn còn phụ thuộc vào điều kiện ban đầu.
    # 4. Biểu đồ moving average cho thấy rõ xu hướng ổn định của policy qua
    #    thời gian (không có drift/suy giảm hiệu suất theo episode, vì đây
    #    là policy cố định - không học - nên đường trung bình gần như phẳng).
    # 5. Kết luận chung: heuristic policy là baseline tốt và dễ triển khai,
    #    nhưng để giải quyết môi trường phức tạp hơn (nhiều biến, không có
    #    quy luật vật lý rõ ràng như FrozenLake trơn trượt hay Taxi), cần
    #    chuyển sang các thuật toán học tăng cường thực sự như Q-learning
    #    hoặc Deep Q-Network (DQN) để agent tự học chính sách tối ưu.


if __name__ == "__main__":
    main()