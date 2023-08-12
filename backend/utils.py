import csv
import os
import urllib.request
from functools import wraps

import cv2
import numpy as np
from flask import request, redirect, make_response


def decode_image(image_data):
    # Parse the URL data into actual binary data
    with urllib.request.urlopen(image_data) as res:
        jpg_data = res.read()

    # Convert the raw data into an OpenCV image
    np_data = np.frombuffer(jpg_data, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return image


def create_initial_agent_queue(queue):
    tmp_dirs = os.listdir('static/logs')
    tmp_dirs.sort()  # Put the most recent models last in the queue (so they are first used)
    for directory in tmp_dirs:
        files = os.listdir(f'static/logs/{directory}')
        files.sort()
        for file in files:
            if ".zip" in file and "backup" not in file:
                queue.put(f'static/logs/{directory}/{file}')


def inner_content(f):
    """Redirects to home page if request doesn't have right headers.
    Avoids user accessing inner content from browser."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return f(*args, **kwargs)
        return redirect('/')

    return decorated


def error_handler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return make_response('Something went wrong.', 400)

    return decorated


# TODO Can be removed, for testing purposes only
def show_images(queue):
    print(f'#images = {queue.qsize()}')
    while not queue.empty():
        image = queue.get()
        cv2.imshow('image', image)
        cv2.waitKey(100)


# TODO Can be removed, for testing purposes only
def read_experience_logs():
    tmp_dirs = os.listdir('static/logs')
    tmp_dirs.sort()  # Put the most recent models last in the queue (so they are first used)
    for directory in tmp_dirs:
        files = os.listdir(f'static/logs/{directory}')
        for file in files:
            if "experience_log" in file:
                with open(f'static/logs/{directory}/{file}', "r") as csv_file:
                    csv_reader = csv.DictReader(filter(lambda x: not x.startswith("#"), csv_file))
                    for row in csv_reader:
                        print("heart:", row["heart_bpm"])
                        print("song:", row["song_file"])
                        print("action:", row["a"])
                        print("reward:", row["r"])
                        print("next heart:", row["next_heart_bpm"])
                        print("next song:", row["next_song_file"])
                        print("terminated:", row["terminated"])
                        print("truncated:", row["truncated"])
                        print()
