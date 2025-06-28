import os
import traci
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.monitor import Monitor
from opti_env import SumoEnv

# --- CONFIGURATION ---
#390x2 514
MODEL_PATH = "E:/licenta/oras-mediu/2025-06-05-11-45-23/ppo_models_phase2/ppo_sumo_phase2_20250614-174623.zip" 
SUMO_CFG_PATH = "E:/licenta/oras-mediu/2025-06-05-11-45-23/osm.sumocfg"
N_EVALUATION_EPISODES = 1
AGENT_OUTPUT_DIR = "agent_outputs"
# --- END CONFIGURATION ---

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)

    eval_env = SumoEnv(sumo_cfg=SUMO_CFG_PATH, gui=True)
    eval_env = Monitor(eval_env)
    model = MultiOutputPPO.load(MODEL_PATH, env=eval_env)

    all_trip_durations = []
    all_trip_waiting_times = []
    episode_final_lengths = []

    print(f"--- Evaluating Agent: {os.path.basename(MODEL_PATH)} ---")
    
    # We manage the closing of the environment manually in the loop
    # to ensure files are handled correctly.
    for i in range(N_EVALUATION_EPISODES):
        print(f"  Running episode {i + 1}/{N_EVALUATION_EPISODES}...")
        obs, info = eval_env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

        episode_final_lengths.append(info['episode']['l'])
        
        # --- ROBUST FILE HANDLING ---
        # 1. Close the SUMO environment for the episode that just finished.
        #    The `close()` method in your env calls `traci.close()`.
        eval_env.env.close()
        
        # 2. THE FIX: Wait for a moment to let the OS release the file lock.
        time.sleep(0.5)

        # 3. Now that the process is closed and we've waited, safely move and parse the file.
        source_path = "tripinfo.xml"
        dest_path = os.path.join(AGENT_OUTPUT_DIR, f"tripinfo_agent_ep{i+1}.xml")
        
        try:
            shutil.move(source_path, dest_path)
            tree = ET.parse(dest_path)
            root = tree.getroot()
            if root.find('tripinfo') is not None:
                all_trip_durations.extend([float(trip.get('duration')) for trip in root.findall('tripinfo')])
                all_trip_waiting_times.extend([float(trip.get('waitingTime')) for trip in root.findall('tripinfo')])
            else:
                print(f"    Info: {os.path.basename(dest_path)} was empty (0 completed trips).")
        except (FileNotFoundError, ET.ParseError) as e:
            print(f"    Warning: Could not process {source_path} for episode {i+1}. Reason: {e}")

    # --- Final Analysis (remains the same) ---
    if not all_trip_durations:
        print("\nNo trip data was collected. Cannot generate summary.")
    else:
        df = pd.DataFrame({'duration': all_trip_durations, 'waitingTime': all_trip_waiting_times})
        print("\n--- Overall Agent Performance Summary ---")
        print(f"Data from {len(all_trip_durations)} trips across {N_EVALUATION_EPISODES} completed episodes with trip data.")
        print(f"Average Episode Length: {np.mean(episode_final_lengths):.2f} steps")
        print("\n--- Trip Duration Stats ---")
        print(df['duration'].describe(percentiles=[.5, .75, .95]))
        print("\n--- Waiting Time Stats ---")
        print(df['waitingTime'].describe(percentiles=[.5, .75, .95]))