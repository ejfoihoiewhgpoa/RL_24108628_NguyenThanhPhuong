import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


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


def compute_returns(rewards, gamma=1.0):
    returns = [0] * len(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


env = gym.make("Blackjack-v1")

def random_policy(state):
    return env.action_space.sample()

N_EPISODES = 5000
gammas = [0.5, 0.8, 0.9, 0.99, 1.0]

# Sinh sẵn N_EPISODES episode, dùng chung cho mọi gamma để so sánh công bằng
episodes = [generate_episode(env, random_policy) for _ in range(N_EPISODES)]

mean_G0_per_gamma = {}
for gamma in gammas:
    G0_list = []
    for episode in episodes:
        rewards = [r for (s, a, r) in episode]
        returns = compute_returns(rewards, gamma=gamma)
        G0_list.append(returns[0])
    mean_G0_per_gamma[gamma] = np.mean(G0_list)

print(f"Số episode: {N_EPISODES}")
print(f"{'gamma':<10}{'mean G_0':<15}")
print("-" * 25)
for gamma in gammas:
    print(f"{gamma:<10}{mean_G0_per_gamma[gamma]:<15.4f}")

mean_G0_values = [mean_G0_per_gamma[g] for g in gammas] 
plt.figure(figsize=(8, 5)) 
plt.plot(gammas, mean_G0_values, marker="o", linewidth=2) 
plt.title("Mean G_0 theo gamma (random policy)") 
plt.xlabel("gamma") 
plt.ylabel("mean G_0") 
plt.grid(True, alpha=0.3) 
plt.savefig("mean_G0_vs_gamma.png", dpi=150, bbox_inches="tight") 
plt.show()

env.close()