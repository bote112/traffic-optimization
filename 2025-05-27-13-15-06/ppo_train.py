import os
import time
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sumo_env import SumoEnv

# === Config ===
SUMO_CFG_PATH = "E:/licenta/oras-mediu/2025-05-27-13-15-06/osm.sumocfg"
LOG_DIR = "E:/licenta/oras-mediu/2025-05-27-13-15-06/ppo_logs"
MODEL_DIR = "E:/licenta/oras-mediu/2025-05-27-13-15-06/ppo_models"
TIMESTEPS = 2_000_000
SAVE_FREQ = 25_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Custom Callback for Autosaving ===
class SaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_file = os.path.join(self.save_path, f"ppo_sumo_{timestamp}")
            self.model.save(model_file)
            if self.verbose > 0:
                print(f"\n[AutoSave] Saved model at step {self.num_timesteps} to: {model_file}")
        return True

# === Custom TensorBoard Logging Callback ===
class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [None])[0]
        if reward is not None:
            self.logger.record("env/reward", reward)

        # If your env returns episode_info (via Monitor), these will appear too:
        return True

# === Create Environment ===
def make_env():
    env = SumoEnv(sumo_cfg=SUMO_CFG_PATH, gui=False)
    env = Monitor(env)  # Required for TensorBoard to track episode rewards
    return env

env = DummyVecEnv([make_env])

# === Create and Train PPO Model ===
model = MultiOutputPPO(
    policy="MultiOutputPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# Combine callbacks
callback = CallbackList([
    SaveCallback(SAVE_FREQ, MODEL_DIR),
    TensorboardLoggingCallback()
])

print("Starting PPO training...")
model.learn(
    total_timesteps=TIMESTEPS,
    callback=callback,
)
print("Training complete.")

# === Final Save ===
final_timestamp = time.strftime("%Y%m%d-%H%M%S")
final_model_path = os.path.join(MODEL_DIR, f"ppo_sumo_final_{final_timestamp}")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
