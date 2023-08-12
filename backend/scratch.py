import datetime
import os
import threading
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

LOG_DIR = 'static/logs'


class ClientThread(threading.Thread):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.callback = None

    def run(self):
        # Create log dir
        log_dir = f'{LOG_DIR}/log{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/'
        os.makedirs(log_dir, exist_ok=True)

        # Create the callback: save every 20 steps
        # (ranges anywhere between 0 and 7 minutes depending on agent's actions)
        self.callback = CancellableCheckpointCallback(
            save_freq=330,
            save_path='static/logs/tests/',
            name_prefix="backup",
            save_vecnormalize=True,
        )

        # Train the agent
        agent = PPO("MlpPolicy", "Pendulum-v1", n_steps=32, batch_size=16)
        agent.learn(100000, callback=self.callback)

        # What needs to happen after stopping
        # # Stop the experiment
        # env.close()  # Puts lasts "end" action in action queue (to stop the experiment client-side)
        #
        # # Save the agent and put the file on the stack for future use
        # agent.save(os.path.join(log_dir, "last_model"))
        # self.agent_queue.put(os.path.join(log_dir, "last_model")+".zip")
        print("yaya")

    def halt_learning(self):
        if self.callback is not None:
            self.callback.stop()


class CancellableCheckpointCallback(CheckpointCallback):
    def __init__(self,
                 save_freq: int,
                 save_path: str,
                 name_prefix: str = "rl_model",
                 save_replay_buffer: bool = False,
                 save_vecnormalize: bool = False,
                 verbose: int = 0):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix,
                         save_replay_buffer=save_replay_buffer, save_vecnormalize=save_vecnormalize, verbose=verbose)
        self.stop_requested = False
        self.counter = 0

    def _on_step(self) -> bool:
        self.counter += 1
        print("on step", self.counter, self.stop_requested)
        if self.stop_requested:
            return False
        return super()._on_step()

    def stop(self):
        self.stop_requested = True


if __name__ == '__main__':
    client = ClientThread(user_id=11)
    client.start()
    time.sleep(1)
    client.halt_learning()

