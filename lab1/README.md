# Bài thực hành số 1
Họ tên: Nguyễn Thanh Phương
MSSV: 24108628
Lớp: N01
GitHub username:RL_24108628_NguyenThanhPhuong
Repository URL: https://github.com/ejfoihoiewhgpoa/RL_24108628_NguyenThanhPhuong
Python version: 3.14.2
Gymnasium version: 1.3.0
NumPy version: 2.5.2
Matplotlib version: 3.11.1
## Cách cài đặt
## Cách chạy từng bài
## Cách chạy chương trình tổng hợp
## Mô tả kết quả
## Khó khăn gặp phải
## Kết luận

## Cách cài đặt

   ### Bước 1: Clone repository về máy
```bash
git clone https://github.com/ejfoihoiewhgpoa/RL_24108628_NguyenThanhPhuong.git
cd RL_24108628_NguyenThanhPhuong
```
   ### Bước 2: Kiểm tra phiên bản Python
Yêu cầu Python >= 3.9. Kiểm tra bằng lệnh:
```bash
python --version
```
   ### Bước 3: (Khuyến nghị) Tạo môi trường ảo để tránh xung đột thư viện
```bash
python -m venv venv
```
Kích hoạt môi trường ảo:
- Windows:
```bash
  venv\Scripts\activate
```
- Linux/Mac:
```bash
  source venv/bin/activate
```

   ### Bước 4: Cài các thư viện cần thiết
```bash
pip install gymnasium[classic-control] numpy matplotlib
```
Giải thích:
- `gymnasium[classic-control]` — thư viện môi trường RL (CartPole, FrozenLake...), phần `classic-control` bao gồm `pygame` để hiển thị hình ảnh mô phỏng.
- `numpy` — tính toán số liệu (mean, std, reward...).
- `matplotlib` — vẽ biểu đồ kết quả.

   ### Bước 5: (Tuỳ chọn) Nếu muốn chạy bằng Jupyter Notebook
```bash
pip install jupyter ipykernel nbconvert
```

   ### Bước 6: Kiểm tra cài đặt thành công
```bash
python -c "import gymnasium, numpy, matplotlib; print('OK')"
```
Nếu in ra `OK` là đã cài đặt xong, sẵn sàng chạy các bài trong repo.

## Cách chạy từng bài

> Các bài 1–35: xem hướng dẫn chạy trong README riêng của từng thư mục bài (nếu có), 
> hoặc bổ sung khi hoàn thành.

   ### Bài 1 
   ### Bài 36 (lab1 — CartPole-v1)
```bash
cd Lab01/src
python3 mini_project_cartpole.py
```
Chương trình sẽ:
1. Tạo môi trường `CartPole-v1`
2. Chạy 500 episode với policy heuristic (dựa vào góc nghiêng pole)
3. In ra thống kê: mean/std/min/max reward, mean length
4. Lưu 2 biểu đồ vào thư mục `Lab01/figures/`
5. Đóng môi trường bằng `env.close()`

## Cách chạy chương trình tổng hợp
```bash
cd lab1/src && python3 mini_project_cartpole.py
```

## Mô tả kết quả
Policy heuristic đạt mean_reward ~42, dao động 24–68. Kết quả ổn định nhưng chưa tối ưu.

## Khó khăn gặp phải
- Nhầm API Gym cũ (done, reset không trả info).
- Thiếu pygame khi render, phải cài gymnasium[classic-control].
- CartPole-v0 deprecated, phải đổi sang v1.

## Kết luận
Pipeline chạy đúng với API mới của Gymnasium. Policy đơn giản cho kết quả trung bình, có thể cải thiện bằng Q-learning ở bài sau.