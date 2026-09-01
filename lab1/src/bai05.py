import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

print("Observation:", observation)
print("Type:", type(observation))
print("Shape:", observation.shape)
print("Info:", info)

for i, value in enumerate(observation):
    # value: từng phần tử lấy ra từ mảng numpy -> kiểu numpy.float32
    print(f"[{i}] value={value}, type={type(value)}")