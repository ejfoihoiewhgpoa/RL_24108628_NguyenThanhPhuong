def compute_returns(rewards, gamma=1.0):
    returns = [0] * len(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# Episode giả
rewards = [0, 0, 1]

# Tính return với các giá trị gamma khác nhau
for gamma in [1.0, 0.9, 0.5]:
    returns = compute_returns(rewards, gamma=gamma)
    print(f"gamma = {gamma}")
    print(f"returns = {returns}")
    for t, (r, g) in enumerate(zip(rewards, returns)):
        print(f"t={t}: reward={r}, G_{t}={g}")