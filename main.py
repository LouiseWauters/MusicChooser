from music_env import MusicEnv


def env_test():
    env = MusicEnv()
    env.reset()
    env.render()
    done = False
    import random
    while not done:
        action = random.randint(0, 1)
        obs, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    env_test()
