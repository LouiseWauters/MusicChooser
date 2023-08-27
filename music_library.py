import os
import random

from song_bpm_utils import SONG_DIRECTORY


class MusicLibrary:
    def __init__(self, repeat_shuffle=False):
        self.song_list = []
        song_categories = os.listdir(SONG_DIRECTORY)
        for category in song_categories:
            self.song_list += [f'{category}/{song}' for song in os.listdir(f'{SONG_DIRECTORY}/{category}')]
        self.repeat_shuffle = repeat_shuffle
        self.picked_songs = set()

    def get_random_song(self):
        if self.repeat_shuffle:
            random_song = random.choice(self.song_list)
        else:
            random_song = random.choice([song for song in self.song_list if song not in self.picked_songs])
            self.picked_songs.add(random_song)
            # Clear picked song list if all songs have been picked once
            if len(self.picked_songs) == len(self.song_list):
                self.picked_songs = set()
        return random_song
