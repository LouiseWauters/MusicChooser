import os
import random

import vlc

from song_bpm_utils import get_bpm


class MusicPlayer:
    def __init__(self, repeat_shuffle=False):
        self.player = vlc.MediaPlayer()
        self.song_list = os.listdir('songs')
        self.media = None
        self.repeat_shuffle = repeat_shuffle
        self.picked_songs = set()

    def set_song(self, song):
        self.media = vlc.Media(f'songs/{song}')
        self.player.set_media(self.media)

    def get_random_song(self):
        if self.repeat_shuffle:
            random_song = random.choice(self.song_list)
        else:
            random_song = random.choice([song for song in self.song_list if song not in self.picked_songs])
            self.picked_songs.add(random_song)
            # Clear picked song list if all songs have been picked once
            if len(self.picked_songs) == len(self.song_list):
                self.picked_songs = set()
        song_bpm = get_bpm(random_song)
        return random_song, song_bpm

    def play(self):
        self.player.play()

    def stop(self):
        self.player.stop()
