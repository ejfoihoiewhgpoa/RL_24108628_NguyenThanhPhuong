import gymnasium as gym


def generate_actions(seed, num_actions=20):
    env = gym.make("CartPole-v1")

    env.action_space.seed(seed)

    actions = []

    for i in range(20):
        action = env.action_space.sample()
        actions.append(action)

    env.close()

    return actions


def main():
    actions_1 = generate_actions(seed=42)

    actions_2 = generate_actions(seed=42)

    print("Chuỗi action lần 1:")
    print(actions_1)

    print("\nChuỗi action lần 2:")
    print(actions_2)

    print("\nHai chuỗi action có giống nhau không?")
    print(actions_1 == actions_2)


if __name__ == "__main__":
    main()