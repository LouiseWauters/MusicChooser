import math
import os
import time

import gymnasium as gym
import numpy as np

from image_bpm import ImageBPM
from music_library import MusicLibrary
from song_bpm_utils import compute_features, SONG_DIRECTORY


class MusicEnv(gym.Env):
    def __init__(self, image_queue, song_queue, max_song_bpm=300, max_heart_bpm=300, goal_heart_bpm=60, max_steps=1000,
                 songs_per_episode=10, song_duration_seconds=20, sampling_rate=22050, hop_length=512, log_dir=None):
        self.song_queue = song_queue
        self.actions = [self.pick_yes, self.pick_no]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.max_song_bpm = max_song_bpm
        self.max_heart_bpm = max_heart_bpm
        self.log_dir = log_dir
        self.sr = sampling_rate  # sampling rate songs
        self.hop_length = hop_length  # hop length of the time series for chroma
        self.chroma_padding = 30
        chroma_length = math.ceil((song_duration_seconds * self.sr) / self.hop_length)
        self.observation_space = gym.spaces.Dict({'heart_bpm': gym.spaces.Discrete(300),
                                                  'song_bpm': gym.spaces.Discrete(300),
                                                  'chroma_stft': gym.spaces.Box(low=0, high=255,
                                                                                shape=(12+self.chroma_padding,
                                                                                       chroma_length, 1),
                                                                                dtype=np.uint8)})
        self.log = ''
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.songs_per_episode = songs_per_episode
        self.songs_left = songs_per_episode
        self.state = None
        self.goal = goal_heart_bpm
        self.music_library = MusicLibrary()
        self.next_song = None
        self.song_duration_seconds = song_duration_seconds
        self.images_to_bpm = ImageBPM(image_queue=image_queue)
        self.images_to_bpm.start()
        self.experience_log = (f'#{{"max_song_bpm": {max_song_bpm}, "max_heart_bpm": {max_heart_bpm}'
                               f', "goal_heart_bpm": {goal_heart_bpm}, "max_steps": {max_steps}'
                               f', "songs_per_episode": {songs_per_episode}'
                               f', "song_duration_seconds": {song_duration_seconds}, "sampling_rate": {sampling_rate}'
                               f', "hop_length": {hop_length}}}\n')
        self.experience_log += "heart_bpm,song_file,a,r,next_heart_bpm,next_song_file,terminated,truncated\n"

    def reset(self, seed=None, options=None):
        self.state = dict()
        self.songs_left = self.songs_per_episode + 1  # default start step takes 1 off
        self.next_song = 'start'  # starting action
        self.pick_yes()
        self.steps_left = self.max_steps
        self.songs_left = self.songs_per_episode
        self.log = self.state_representation()
        return self.state.copy(), {}

    def step(self, action: int):
        previous_state = self.state.copy()
        self.experience_log += f'{self.state["heart_bpm"]},"{self.next_song}",{action},'

        # Do selected action
        self.actions[action]()

        # Calculate reward
        reward = distance(previous_state["heart_bpm"], self.goal) - distance(self.state["heart_bpm"], self.goal)

        terminated = False

        if self.songs_left == 0:
            terminated = True
            self.log += 'End of episode.\n'
            reward = 100 - distance(self.state["heart_bpm"], self.goal)

        self.steps_left -= 1
        truncated = self.steps_left <= 0

        if truncated:
            reward = -1000

        # if terminated or truncated:
        #     self.song_queue.put('end')

        self.log += f'Reward: {reward}\n'
        self.log += self.state_representation(previous_state=previous_state)
        self.experience_log += f'{reward},{self.state["heart_bpm"]},"{self.next_song}",{terminated},{truncated}\n'
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        self.song_queue.put('end')
        self.save_experience_log()

    def render(self, mode=None):
        print(self.log)
        self.log = ''

    def pick_yes(self):
        self.songs_left -= 1
        # Only put the directory before the action if it is a file name
        self.song_queue.put(f'{SONG_DIRECTORY}/{self.next_song}' if "." in self.next_song else self.next_song)
        # Sleep so images can come in
        time.sleep(self.song_duration_seconds)

        # Get new heart bpm
        counter = 0
        while self.images_to_bpm.getBPM() == 60.0:  # Buffer isn't full yet
            time.sleep(0.1)
            counter += 1
            if counter % 100 == 0:
                print("Waiting for more images.")
        new_bpm = self.images_to_bpm.getBPM()
        self.state["heart_bpm"] = int(new_bpm)
        self.set_next_song()

    def pick_no(self):
        self.set_next_song()

    def set_next_song(self):
        self.next_song = self.music_library.get_random_song()
        features = compute_features(self.next_song, duration_seconds=self.song_duration_seconds)
        self.state['song_bpm'] = int(features['bpm'])
        chroma_stft = np.round(features['chroma_stft'] * 255).astype(np.uint8)
        # zero padding at top and bottom
        padding = int(self.chroma_padding / 2)
        chroma_stft = np.concatenate((np.zeros((padding, chroma_stft.shape[1])), chroma_stft), axis=0)
        chroma_stft = np.concatenate((chroma_stft, np.zeros((self.chroma_padding - padding, chroma_stft.shape[1]))),
                                     axis=0)
        self.state['chroma_stft'] = np.expand_dims(chroma_stft, axis=2).astype(np.uint8)

    def state_representation(self, previous_state=None):
        rep_str = ''
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
