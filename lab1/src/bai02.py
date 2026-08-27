import gymnasium as gym

def main():
    env = gym.make("CartPole-v1")
    print(env)

    env.close()
if __name__ == "__main__":
    main()
