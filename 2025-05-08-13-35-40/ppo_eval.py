import os
import numpy as np
from stable_baselines3 import PPO
from sumo_env import SumoEnv

# === Load trained model ===
MODEL_PATH = "E:/licenta/oras-mic/2025-05-08-13-35-40/ppo_models_delay/ppo_delay_20250523-113137"
model = PPO.load(MODEL_PATH)

# === Config ===
SUMO_CFG = "E:/licenta/oras-mic/2025-05-08-13-35-40/osm.sumocfg"
NUM_EPISODES = 500

env = SumoEnv(sumo_cfg=SUMO_CFG, gui=False)

all_waiting_times = []
all_total_halted = []
all_episode_lengths = []

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    steps = 0
    total_wait = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_wait += -reward
        steps += 1

    avg_wait = total_wait / steps
    all_waiting_times.append(avg_wait)
    all_total_halted.append(total_wait)
    all_episode_lengths.append(steps)

    print(f"[PPO Episode {ep + 1}] Steps: {steps}, Avg Wait: {avg_wait:.2f}, Total Halted: {total_wait:.0f}")

env.close()
