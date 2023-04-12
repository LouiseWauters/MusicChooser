import os

import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def __init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f'Num timesteps: {self.num_timesteps}')
                    print(f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}')

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
        return True


def main():
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = MusicEnv()
    env = Monitor(env, log_dir)

    model = PPO("MlpPolicy", env, verbose=1)
    # Create the callback: check every 10 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

    # Train the agent
    timesteps = 1e5
    model.learn(total_timesteps=int(timesteps), callback=callback)

    plot_results([log_dir], int(timesteps), results_plotter.X_EPISODES, "PPO Music Chooser")
    plt.show()

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f'Reward per episode {mean_reward} +- {std_reward}')

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
