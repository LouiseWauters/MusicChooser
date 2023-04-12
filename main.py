import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

from music_env import MusicEnv


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
