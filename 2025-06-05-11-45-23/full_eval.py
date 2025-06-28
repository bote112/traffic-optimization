import os
import subprocess
import shutil
import time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.monitor import Monitor
from opti_env import SumoEnv # Assumes your SumoEnv class is in this file

# --- CONFIGURATION ---
# Base directory where all your simulation files are located
BASE_DIR = "E:/licenta/oras-mediu/2025-06-05-11-45-23"

# Path to your trained agent model
MODEL_PATH = os.path.join(BASE_DIR, "ppo_models_phase2", "ppo_sumo_phase2_20250615-212416.zip")

# Number of different scenarios to evaluate
N_EVALUATION_EPISODES = 25

# Directory to store all output files
OUTPUT_DIR = "evaluation_run_outputs"
BASELINE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "baseline_outputs")
AGENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "agent_outputs")
# --- END CONFIGURATION ---


def generate_scenario():
    #This runs your python generator scripts and duarouter.
    subprocess.run(["python", os.path.join(BASE_DIR, "simulation_generator.py"),
                    os.path.join(BASE_DIR, "full.rou.xml")], check=True)    
    subprocess.run(["python", os.path.join(BASE_DIR, "truck_generator.py")], check=True)

    duarouter_cmd = [
        "duarouter",
        "-n", os.path.join(BASE_DIR, "osm.net.xml"),
        "--trip-files", os.path.join(BASE_DIR, "trips.pedestrians.xml"),
        "-a", os.path.join(BASE_DIR, "vtypes.xml"),
        "-o", os.path.join(BASE_DIR, "ped.rou.xml"),
    ]

    print("  -> Running duarouter...")
    result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] duarouter failed:")
        print(result.stderr)
        # Exit the script if we can't generate a scenario
        exit()
    print("  -> Scenario generated successfully.")


def run_baseline_episode(sumo_cfg_path, output_xml_path):
    """
    Runs a single SUMO episode using the default traffic light logic (baseline).
    """    
    sumo_cmd = [
        "sumo", # Use 'sumo' for no GUI, which is faster
        "-c", sumo_cfg_path,
        "--tripinfo-output", output_xml_path,
        "--no-step-log", "true", # Suppress verbose logging
    ]
    
    subprocess.run(sumo_cmd, check=True)
    print("  -> Baseline episode finished.")

def run_agent_episode(model, env, output_xml_path):
    """
    Runs a single SUMO episode controlled by the RL agent.
    """

    # Set the tripinfo output file within the environment's config
    # This requires a small modification to your SumoEnv to handle this.
    # For now, we rely on the default name and move the file after.
    obs, info = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Close the environment to release file locks
    env.env.close()
    time.sleep(0.5) # Give the OS a moment to release the file

    # Move the generated tripinfo file to its final destination
    default_tripinfo_path = "tripinfo.xml"
    if os.path.exists(default_tripinfo_path):
        shutil.move(default_tripinfo_path, output_xml_path)

    print("  -> Agent episode finished.")


def parse_trip_data(file_path):
    """
    Parses a tripinfo XML file and returns lists of durations and waiting times.
    """
    durations, waiting_times = [], []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for trip in root.findall('tripinfo'):
            durations.append(float(trip.get('duration')))
            waiting_times.append(float(trip.get('waitingTime')))

    except (FileNotFoundError, ET.ParseError) as e:
        print(f"    Warning: Could not process {os.path.basename(file_path)}. Reason: {e}")
    
    return durations, waiting_times

def print_summary(title, all_durations, all_waiting_times):
    """Prints a formatted summary of the collected statistics."""
    if not all_durations:
        print(f"\n--- {title} ---")
        print("No trip data was collected.")
        return

    df = pd.DataFrame({'duration': all_durations, 'waitingTime': all_waiting_times})
    
    print(f"\n{'-'*20} {title.upper()} PERFORMANCE SUMMARY {'-'*20}")
    print(f"Data from {len(df)} total trips across {N_EVALUATION_EPISODES} episodes.")
    
    print("\n--- Trip Duration Stats (seconds) ---")
    print(df['duration'].describe(percentiles=[.5, .75, .95, .99]))
    
    print("\n--- Waiting Time Stats (seconds) ---")
    print(df['waitingTime'].describe(percentiles=[.5, .75, .95, .99]))
    print(f"{'-'* (44 + len(title))}\n")


if __name__ == "__main__":
    # --- SETUP ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)

    # Initialize the environment and load the model once
    sumo_cfg = os.path.join(BASE_DIR, "osm.sumocfg")
    agent_env = SumoEnv(sumo_cfg=sumo_cfg, gui=False)
    agent_env = Monitor(agent_env)
    model = MultiOutputPPO.load(MODEL_PATH, env=agent_env)

    # Data storage
    all_baseline_durations, all_baseline_waits = [], []
    all_agent_durations, all_agent_waits = [], []

    # --- MAIN EVALUATION LOOP ---
    for i in range(N_EVALUATION_EPISODES):
        # 1. Generate the unique scenario for this episode
        generate_scenario()

        # 2. Run Baseline
        baseline_output_file = os.path.join(BASELINE_OUTPUT_DIR, f"tripinfo_baseline_ep{i+1}.xml")
        run_baseline_episode(sumo_cfg, baseline_output_file)
        durations, waits = parse_trip_data(baseline_output_file)
        all_baseline_durations.extend(durations)
        all_baseline_waits.extend(waits)

        # 3. Run Agent
        agent_output_file = os.path.join(AGENT_OUTPUT_DIR, f"tripinfo_agent_ep{i+1}.xml")
        run_agent_episode(model, agent_env, agent_output_file)
        durations, waits = parse_trip_data(agent_output_file)
        all_agent_durations.extend(durations)
        all_agent_waits.extend(waits)

    # --- FINAL RESULTS ---
    print_summary("Baseline", all_baseline_durations, all_baseline_waits)
    print_summary("Agent", all_agent_durations, all_agent_waits)

