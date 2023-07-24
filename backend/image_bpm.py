import numpy as np
import cv2
import threading
import collections

WIDTH = 640
HEIGHT = 480
LEVELS = 3
BUFFER_SIZE = 150
FPS = 10


class ImageBPM(threading.Thread):
    def __init__(self, image_queue):
        super().__init__()
        self.lock = threading.Lock()

        self.image_queue = image_queue

        self.gauss = collections.deque([], maxlen=BUFFER_SIZE)
        self.bpm = 60.0

        self.values = []

    def run(self):
        # Bandpass Filter for Specified Frequencies
        frequencies = FPS * np.arange(BUFFER_SIZE) / BUFFER_SIZE
        mask = (frequencies >= 0.8) & (frequencies <= 2.5)          # 0.8 Hz to 2.5 Hz = 48 to 150 bpm

        while True:
            frame = self.image_queue.get()

            # Construct Gaussian Pyramid
            self.gauss.append(self.buildGauss(frame, LEVELS+1)[-1])

            if len(self.gauss) < BUFFER_SIZE:
                continue

            # Bandpass Filter
            fourierTransform = np.real(np.fft.fft(self.gauss, axis=0))
            fourierTransform[~mask] = 0

            # Grab a Pulse
            avg_per_timestep = [f.mean() for f in fourierTransform]
            instant_hz = frequencies[np.argmax(avg_per_timestep)]
            instant_bpm = 60.0 * instant_hz

            with self.lock:
                self.bpm = 0.95 * self.bpm + 0.05 * instant_bpm
                self.values.append(self.bpm)
                # print('BPM', self.bpm)

    def getBPM(self):
        with self.lock:
            return self.bpm

    # Helper Methods
    def buildGauss(self, frame, levels):
        pyramid = [frame]

        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)

        return pyramid


if __name__ == '__main__':
    t = ImageBPM()
    t.start()

    import time
    time.sleep(50.0)
