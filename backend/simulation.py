import os
import random

import gymnasium as gym
from stable_baselines3 import PPO
import os
import random

import gymnasium as gym
from stable_baselines3 import PPO


class HeartRateSimulation:
    def __init__(self):
        self.min = 60
        self.max = 150

    def get_new_rate(self, start, change):
        change_var = change + (random.random() * 2) - 1  # variance of 1
        if change_var < 0:
            alpha = (start - self.min) / (self.max - self.min) * 0.8
        else:
            alpha = (start - self.max) / (self.min - self.max) * 0.5
        new_rate = start + alpha * change_var
        return max(self.min, min(self.max, new_rate))


class MusicEnv(gym.Env):
    def __init__(self, goal_heart_bpm=60, max_steps=1000, songs_per_episode=10, log_dir=None):
        self.song_options = {
            'rock': 20,
            'salsa': 15,
            'classic': -15,
            'meditation': -20,
            'techno': 30
        }
        self.heart = HeartRateSimulation()
        self.heart_rate = 80
        self.actions = [self.pick_yes, self.pick_no]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Dict({'heart_bpm': gym.spaces.Discrete(300),
                                                  'song_bpm': gym.spaces.Discrete(300)})
        self.log = ''
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.songs_per_episode = songs_per_episode
        self.songs_left = songs_per_episode
        self.state = None
        self.goal = goal_heart_bpm
        self.next_song = None
        self.experience_log = ("heart_bpm,song_file,song_bpm,a,r,next_heart_bpm,next_song_file,next_song_bpm"
                               ",terminated,truncated\n")

    def reset(self, seed=None, options=None):
        self.state = dict()
        self.set_next_song()
        self.state["heart_bpm"] = int(self.heart_rate)
        self.steps_left = self.max_steps
        self.songs_left = self.songs_per_episode
        self.log = self.state_representation()
        return self.state.copy(), {}

    def step(self, action: int):
        previous_state = self.state.copy()
        self.experience_log += f'{self.state["heart_bpm"]},"{self.next_song}",{self.state["song_bpm"]},{action},'

        # Do selected action
        self.actions[action]()
        self.log += f'Action: {"yes" if action == 0 else "no"}\n'

        # Calculate reward
        reward = distance(previous_state["heart_bpm"], self.goal) - distance(self.state["heart_bpm"], self.goal)

        terminated = False

        if self.songs_left == 0:
            terminated = True
            reward = 100 - distance(self.heart_rate, self.goal)

        self.steps_left -= 1
        truncated = self.steps_left <= 0
        if truncated:
            reward = -1000

        self.log += f'Reward: {reward}\n\n'

        if terminated:
            self.log += 'End of episode.\n'

        self.log += self.state_representation(previous_state=previous_state)
        self.experience_log += (f'{reward},{self.state["heart_bpm"]},"{self.next_song}",{self.state["song_bpm"]}'
                                f',{terminated},{truncated}\n')
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        self.save_experience_log()

    def render(self, mode=None):
        print(self.log, end='')
        self.log = ''

    def pick_yes(self):
        self.songs_left -= 1
        self.set_new_heart_rate()
        self.set_next_song()

    def pick_no(self):
        self.set_next_song()

    def set_next_song(self):
        self.next_song = random.choice(list(self.song_options.keys()))
        self.state["song_bpm"] = bpm_from_change(self.song_options[self.next_song])

    def set_new_heart_rate(self):
        self.heart_rate = self.heart.get_new_rate(self.heart_rate, self.song_options[self.next_song])
        self.state["heart_bpm"] = int(self.heart_rate)

    def state_representation(self, previous_state=None):
        rep_str = ''
        # for observation in ['song_bpm']:
        for observation in ['song_bpm', 'heart_bpm']:
            rep_str += f'{observation}: {self.state[observation]} BPM '
            if previous_state is not None:
                rep_str += '↑' if self.state[observation] > previous_state[observation] \
                    else '↓' if self.state[observation] < previous_state[observation] \
                    else '-'
            rep_str += '\n'
        return rep_str

    def save_experience_log(self):
        if self.log_dir is not None:
            filename = os.path.join(self.log_dir, "experience_log.csv")
            with open(filename, 'w') as file:
                file.write(self.experience_log)


def distance(x, y):
    return abs(x - y)


def bpm_from_change(change):
    change_min = -50
    change_max = 50
    bpm = (change - change_min) / (change_max - change_min) * 250 + 25
    bpm += (random.random() * 50) - 25
    return int(bpm)


def ppo_test():
    for i in range(10):
        log_dir = f'static/logs/tests/{i}/'
        os.makedirs(log_dir, exist_ok=True)
        env = MusicEnv(log_dir=log_dir)
        agent = PPO("MultiInputPolicy", env=env, n_steps=32, batch_size=16, tensorboard_log=log_dir)
        agent.learn(100000)
        agent.save(os.path.join(log_dir, "last_model"))
        env.close()


if __name__ == '__main__':
    ppo_test()








