import gym
import numpy as np


class MusicEnv(gym.Env):
    def __init__(self, song_bpm_low=10, song_bpm_high=300, heart_bpm_low=30, heart_bpm_high=200, heart_bpm_goal=100,
                 max_steps=1000):
        self.actions = [self.pick_yes, self.pick_no]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.song_bpm_range = (song_bpm_low, song_bpm_high)
        self.heart_bpm_range = (heart_bpm_low, heart_bpm_high)
        self.observations = ['song_bpm', 'heart_bpm']
        self.observation_space = gym.spaces.MultiDiscrete([song_bpm_high - song_bpm_low,
                                                           heart_bpm_high - heart_bpm_low])
        self.log = ''
        self.max_steps = max_steps
        self.steps_left = max_steps
        self.state = None
        self.goal = heart_bpm_goal
        self.danger_low = 50
        self.danger_high = 160

    def observation(self):
        return np.array([self.state['song_bpm'] - self.song_bpm_range[0],
                         self.state['heart_bpm'] - self.heart_bpm_range[0]])

    def reset(self):
        self.state = {'song_bpm': random_bpm(self.song_bpm_range), 'heart_bpm': 60}
        self.steps_left = self.max_steps
        self.log = state_representation(state_to_array(self.state))
        return self.observation()

    def step(self, action: int):
        old_state = self.observation()

        # Do selected action
        self.actions[action]()
        self.log += f'Chosen action: {self.actions[action].__name__}\n'

        new_state = self.observation()

        transformed_goal = self.goal - self.heart_bpm_range[0]

        reward = -1
        done = False

        if distance(transformed_goal, new_state[1]) > distance(transformed_goal, old_state[1]):
            # Heart rate going in the wrong direction
            reward = -10
            self.log += 'Heart rate going in the wrong direction.\n'

        if distance(transformed_goal, new_state[1]) < 10:
            reward = 0
            done = True
            self.log += 'Goal rate reached.\n'

        if self.state['heart_bpm'] >= self.danger_high or self.state['heart_bpm'] <= self.danger_low:
            reward = -1000
            self.state['heart_bpm'] = 60
            self.log += 'User saw god for a second there.\n'

        self.log += state_representation(state_to_array(self.state), (old_state[0] + self.song_bpm_range[0], old_state[1] + self.heart_bpm_range[0]))

        self.steps_left -= 1
        done = done or self.steps_left <= 0
        return self.observation(), reward, done, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''

    def pick_yes(self):
        new_heart_bpm = int(self.state['heart_bpm'] + 0.1 * (self.state['song_bpm'] - self.state['heart_bpm']))
        new_heart_bpm += np.random.randint(-5, 6)
        self.state['heart_bpm'] = min(self.heart_bpm_range[1], new_heart_bpm)
        self.state['heart_bpm'] = max(self.heart_bpm_range[0], self.state['heart_bpm'])
        self.state['song_bpm'] = random_bpm(self.song_bpm_range)

    def pick_no(self):
        self.state['song_bpm'] = random_bpm(self.song_bpm_range)


def random_bpm(bpm_range):
    return np.random.randint(bpm_range[0], bpm_range[1])


def state_representation(state, previous_state=None):
    rep_str = f'song: {state[0]} BPM '
    if previous_state is not None:
        rep_str += '↑' if state[0] > previous_state[0] else '↓' if state[0] < previous_state[0] else '-'
    rep_str += '\n'
    rep_str += f'heart: {state[1]} BPM '
    if previous_state is not None:
        rep_str += '↑' if state[1] > previous_state[1] else '↓' if state[1] < previous_state[1] else '-'
    rep_str += '\n'
    return rep_str


def state_to_array(state):
    return np.array([state[observation] for observation in ['song_bpm', 'heart_bpm']])


def distance(x, y):
    return abs(x - y)