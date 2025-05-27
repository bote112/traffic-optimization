import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from sumo_env import SumoEnv

# === Config ===
SUMO_CFG_PATH = "E:/licenta/oras-mic/2025-05-08-13-35-40/osm.sumocfg"
LOG_DIR = "./ppo_logs"
MODEL_DIR = "./ppo_models"
TIMESTEPS = 1_000_000  # Target total timesteps
SAVE_FREQ = 25_000     # Save every 25k timesteps

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Custom Callback for Autosaving ===
class SaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_file = os.path.join(self.save_path, f"ppo_sumo_{timestamp}")
            self.model.save(model_file)
            if self.verbose > 0:
                print(f"\n[AutoSave] Saved model at step {self.num_timesteps} to: {model_file}")
        return True

# === Create Environment ===
def make_env():
    return SumoEnv(sumo_cfg=SUMO_CFG_PATH, gui=False)

env = make_vec_env(make_env, n_envs=1)

# === Create and Train PPO Model ===
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

print("Starting PPO training...")
model.learn(
    total_timesteps=TIMESTEPS,
    callback=SaveCallback(SAVE_FREQ, MODEL_DIR)
)
print("Training complete.")

# === Final Save ===
final_timestamp = time.strftime("%Y%m%d-%H%M%S")
final_model_path = os.path.join(MODEL_DIR, f"ppo_sumo_final_{final_timestamp}")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
