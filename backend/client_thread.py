import datetime
import os
import queue
import threading

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from music_env import MusicEnv

LOG_DIR = 'static/logs'


class ClientThread(threading.Thread):
    def __init__(self, user_id, image_queue, action_queue, agent_queue):
        super().__init__()
        self.user_id = user_id
        self.image_queue = image_queue
        self.action_queue = action_queue
        self.agent_queue = agent_queue

    def run(self):
        # Create log dir
        log_dir = f'{LOG_DIR}/log{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/'
        os.makedirs(log_dir, exist_ok=True)

        # TODO remove seconds, add some config constant
        # TODO change songs per episode
        # Create and wrap the environment
        env = MusicEnv(image_queue=self.image_queue, song_queue=self.action_queue, song_duration_seconds=20,
                       songs_per_episode=10)
        env = Monitor(env, log_dir)

        try:
            # See if there is a saved agent that can be loaded
            old_agent_file = self.agent_queue.get_nowait()
            agent = PPO.load(old_agent_file, env=env)
        except queue.Empty:
            # If there is no saved agent, make a new agent
            agent = PPO("MultiInputPolicy", env, verbose=1, n_steps=32, batch_size=16, tensorboard_log=log_dir)

        # Create the callback: check every 10 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir, verbose=1)

        # Train the agent
        timesteps = 64
        agent.learn(total_timesteps=int(timesteps), callback=callback, reset_num_timesteps=False)

        # Stop the experiment
        env.close()  # Puts lasts "end" action in action queue (to stop the experiment client-side)

        # Save the agent and put the file on the stack for future use
        agent.save(os.path.join(log_dir, "last_model"))
        self.agent_queue.put(os.path.join(log_dir, "last_model")+".zip")


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
                    # TODO this model could be on the queue as well?
        return True
