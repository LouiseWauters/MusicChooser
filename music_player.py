import os
import random
import time

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

    def play(self, max_seconds_to_play=None):
        """Plays loaded mp3 for a certain amount of seconds.
        If max_seconds_to_play is None, mp3 file plays completely."""
        self.player.play()
        start_time = time.time()
        time.sleep(1)
        if max_seconds_to_play:
            while time.time() - start_time < max_seconds_to_play and self.player.is_playing():
                time.sleep(0.1)
        else:
            while self.player.is_playing():
                time.sleep(0.1)
        self.stop()

    def stop(self):
        self.player.stop()
