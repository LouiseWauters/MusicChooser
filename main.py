from music_env import MusicEnv


def env_test():
    env = MusicEnv()
    env.reset()
    env.render()
    terminated = False
    truncated = False
    import random
    while not (terminated or truncated):
        action = random.randint(0, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        obs['song_bpm'] = 0
        env.render()


if __name__ == '__main__':
    env_test()
