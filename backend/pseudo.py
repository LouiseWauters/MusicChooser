import urllib.request
import cv2
import numpy as np

# Get the toDataURL data. This part would come from a webserver
with open("/tmp/copied_data.txt", "r") as f:
    url_data = f.read()

# Parse the URL data into actual binary data
with urllib.request.urlopen(url_data) as res:
    jpg_data = res.read()

# Convert the raw data into an OpenCV image
np_data = np.frombuffer(jpg_data, dtype=np.uint8)
image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

cv2.imshow('image', image)
cv2.waitKey(4000)


class MyEnv(gymnasium.Env):
    def __init__(self, image_queue, mp3_queue):
        self.image_queue = image_queue
        self.mp3_queue = mp3_queue

    def reset(self):
        state = self.step(default_action)
        return state

    def step(self, action):
        mp3_filename = self.action_to_filename(action)
        self.mp3_queue.put(mp3_filename)

        # Wait for 10 seconds
        time.sleep(10.0)

        # Get and process the images
        images = []

        while not self.image_queue.empty():
            images.append(self.image_queue.get())

        do_process(images)

        state = ...
        reward = ...
        done = ...

        return state, reward, done, truncated, {}



class ClientThread(threading.Thread):
    def run(self):
        env = MyEnv(image_queues[self.id], mp3_queues[self.id])
        agent = sb3.PPO(env=env)

        agent.learn()


class Server:
    def post_image(self, image, user_id):
        image_queues[user_id].put(decoded_image)
        return ''

    def get_action(self, user_id):
        mp3_filename = mp3_queues[user_id].get()
        return mp3_filename