import os

import librosa
import librosa.display
import librosa.feature
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3

SONG_DIRECTORY = 'static/audio'


def compute_bpm_from_file(song_file_name):
    waveform, sample_rate = librosa.load(f'{SONG_DIRECTORY}/{song_file_name}')
    tempo = compute_bpm(waveform, sample_rate)
    return int(tempo)


def compute_features(song_file_name, duration_seconds):
    y, sr = librosa.load(f'{SONG_DIRECTORY}/{song_file_name}', duration=duration_seconds, offset=0)
    bpm = get_bpm(song_file_name)
    if bpm is None:
        bpm = compute_bpm(y=y, sr=sr)[0]
    # zero_crossing_rate = compute_zero_crossing_rate(y=y)
    # spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    return {'bpm': bpm, 'chroma_stft': chroma_stft}


def compute_bpm(y, sr):
    return librosa.feature.tempo(y=y, sr=sr)


def set_bpm(song, bpm):
    mp3file = MP3(f'{SONG_DIRECTORY}/{song}', ID3=EasyID3)
    mp3file['bpm'] = str(bpm)
    mp3file.save()


def get_bpm(song):
    mp3file = MP3(f'{SONG_DIRECTORY}/{song}', ID3=EasyID3)
    bpm_list = mp3file.get('bpm', None)
    return int(bpm_list[0]) if bpm_list is not None else None


def compute_all_bpm():
    song_list = []
    song_categories = os.listdir(SONG_DIRECTORY)
    for category in song_categories:
        song_list += [f'{category}/{song}' for song in os.listdir(f'{SONG_DIRECTORY}/{category}')]
    for song_file_name in song_list:
        if get_bpm(song_file_name) is None:
            song_bpm = compute_bpm_from_file(song_file_name)
            set_bpm(song_file_name, song_bpm)
