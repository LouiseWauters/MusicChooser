import os

import librosa

from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3


def compute_bpm(song):
    waveform, sample_rate = librosa.load(f'songs/{song}')
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    return int(tempo)


def set_bpm(song, bpm):
    mp3file = MP3(f'songs/{song}', ID3=EasyID3)
    mp3file['bpm'] = str(bpm)
    mp3file.save()


def get_bpm(song):
    mp3file = MP3(f'songs/{song}', ID3=EasyID3)
    bpm_list = mp3file.get('bpm', None)
    return int(bpm_list[0]) if bpm_list is not None else None


def compute_all_bpm():
    songs = os.listdir('songs')
    for song in songs:
        if get_bpm(song) is None:
            song_bpm = compute_bpm(song)
            set_bpm(song, song_bpm)
