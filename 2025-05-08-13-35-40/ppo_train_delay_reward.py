# ppo_train_delay_reward.py

import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sumo_env_delay import SumoEnvDelay

# === Config ===
SUMO_CFG_PATH = "E:/licenta/oras-mic/2025-05-08-13-35-40/osm.sumocfg"
LOG_DIR = "./ppo_logs_delay"
MODEL_DIR = "./ppo_models_delay"
TIMESTEPS = 2_000_000
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Env Setup ===
def make_env():
    return SumoEnvDelay(sumo_cfg=SUMO_CFG_PATH, gui=False)

env = make_vec_env(make_env, n_envs=1)

# === PPO Training ===
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

print("Starting PPO training using delay-based reward...")
model.learn(total_timesteps=TIMESTEPS)

timestamp = time.strftime("%Y%m%d-%H%M%S")
model_path = os.path.join(MODEL_DIR, f"ppo_delay_{timestamp}")
model.save(model_path)
print(f"Model saved to: {model_path}")
