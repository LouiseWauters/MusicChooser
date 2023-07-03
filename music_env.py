import time

import gym
import numpy as np

from music_player import MusicPlayer


class MusicEnv(gym.Env):
    def __init__(self, max_song_bpm=300, max_heart_bpm=300, goal_heart_bpm=100, max_steps=1000, songs_per_episode=10):
        self.actions = [self.pick_yes, self.pick_no]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.max_song_bpm = max_song_bpm
        self.max_heart_bpm = max_heart_bpm
        self.observations = ['song_bpm', 'heart_bpm']
        self.observation_space = gym.spaces.MultiDiscrete([max_song_bpm, max_heart_bpm])
        self.log = ''
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.songs_per_episode = songs_per_episode
        self.songs_left = songs_per_episode
        self.state = None
        self.goal = goal_heart_bpm
        # self.danger_low = 50
        # self.danger_high = 160
        self.music_player = MusicPlayer()
        self.next_song = None

    def observation(self):
        return np.array([self.state[observation] for observation in self.observations])

    def reset(self):
        self.state = {'song_bpm': 0, 'heart_bpm': 60}
        self.set_next_song()
        self.steps_left = self.max_steps
        self.songs_left = self.songs_per_episode
        self.log = self.state_representation()
        return self.observation()

    def step(self, action: int):
        previous_state = self.observation()

        # Do selected action
        self.actions[action]()
        self.log += f'Chosen action: {self.actions[action].__name__}\n'

        reward = - distance(self.goal, self.state['heart_bpm'])
        done = False

        if distance(self.goal, self.state['heart_bpm']) < 5:
            reward = 0
            # done = True
            self.log += 'Goal rate reached.\n'

        if self.songs_left == 0:
            done = True
            self.log += 'End of episode.\n'

        self.steps_left -= 1
        done = done or self.steps_left <= 0

        # TODO Calculate (average?) bpm and fill in state

        self.log += self.state_representation(previous_state=previous_state)
        return self.observation(), reward, done, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''

    def pick_yes(self):
        self.songs_left -= 1
        self.music_player.set_song(self.next_song)
        self.music_player.play(max_seconds_to_play=10)
        self.set_next_song()

        # TODO remove
        new_heart_bpm = int(self.state['heart_bpm'] + 0.1 * (self.state['song_bpm'] - self.state['heart_bpm']))
        new_heart_bpm += np.random.randint(-5, 6)
        self.state['heart_bpm'] = min(self.max_heart_bpm, new_heart_bpm)
        self.state['heart_bpm'] = max(0, self.state['heart_bpm'])

    def pick_no(self):
        self.set_next_song()

    def set_next_song(self):
        self.next_song, song_bpm = self.music_player.get_random_song()
        self.state['song_bpm'] = song_bpm

    def state_representation(self, previous_state=None):
        rep_str = ''
        for i, observation in enumerate(self.observations):
            rep_str += f'{observation}: {self.state[observation]} BPM '
            if previous_state is not None:
                rep_str += '↑' if self.state[observation] > previous_state[i] \
                    else '↓' if self.state[observation] < previous_state[i] \
                    else '-'
            rep_str += '\n'
        return rep_str


def distance(x, y):
    return abs(x - y)
