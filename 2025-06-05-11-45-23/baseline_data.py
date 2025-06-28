import os
import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import subprocess
from torch.utils.tensorboard import SummaryWriter

# --- CONFIGURATION ---
USE_GUI = False
CONFIG_FILE = "osm.sumocfg"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_BINARY = "sumo-gui" if USE_GUI else "sumo" 

N_EVALUATION_EPISODES = 1 
EPISODE_TIMEOUT_STEPS = 3600 

# --- Directory and Logging Setup ---
OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the output directory if it doesn't exist

TENSORBOARD_LOGDIR = os.path.join(BASE_DIR, "tensorboard_baseline")
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Running Baseline Evaluation for {N_EVALUATION_EPISODES} Episodes ---")
    
    for i in range(N_EVALUATION_EPISODES):
        print(f"  Running episode {i + 1}/{N_EVALUATION_EPISODES}...")
        
        tripinfo_filename = os.path.join(OUTPUT_DIR, f"tripinfo_baseline_ep{i}.xml")
        base_dir = os.path.dirname(__file__)


           # Commented out the subprocess calls for generating trips and routes in idea of using pre-generated files for training.


        # subprocess.run(["python", os.path.join(base_dir, "simulation_generator.py"),
        #                 os.path.join(base_dir, "full.rou.xml")], check=True)
        # subprocess.run(["python", os.path.join(base_dir, "truck_generator.py")], check=True)

        # duarouter_cmd = [
        #     "duarouter",
        #     "-n", os.path.join(base_dir, "osm.net.xml"),
        #     "--trip-files", os.path.join(base_dir, "trips.pedestrians.xml"),
        #     "-a", os.path.join(base_dir, "vtypes.xml"),
        #     "-o", os.path.join(base_dir, "ped.rou.xml")
        # ]

        # print("[INFO] Running duarouter for vehicle + pedestrian trips...")
        # result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
        # if result.returncode != 0:
        #     print("[ERROR] duarouter failed:")
        #     print(result.stderr)



        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", tripinfo_filename])
        
        step = 0
        while step < EPISODE_TIMEOUT_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
        
        traci.close()
    
    writer.close()
    print("\n--- All episodes complete. Analyzing results... ---")

    # --- Final Analysis: Read all generated XML files ---
    all_trip_durations = []
    all_trip_waiting_times = []

    for i in range(N_EVALUATION_EPISODES):
        filepath = os.path.join(OUTPUT_DIR, f"tripinfo_baseline_ep{i}.xml")
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            all_trip_durations.extend([float(trip.get('duration')) for trip in root.findall('tripinfo')])
            all_trip_waiting_times.extend([float(trip.get('waitingTime')) for trip in root.findall('tripinfo')])
        except (FileNotFoundError, ET.ParseError):
            print(f"    Warning: Could not parse {filepath}. Skipping file.")
    writer.close()

    # --- Final Analysis over ALL trips from ALL episodes ---
    if not all_trip_durations:
        print("No trip data was collected. Cannot generate summary.")
    else:
        # Using Pandas DataFrame for powerful and easy statistics
        df = pd.DataFrame({
            'duration': all_trip_durations,
            'waitingTime': all_trip_waiting_times
        })

        print("\n--- Overall Baseline Performance Summary ---")
        print(f"Data collected from {len(all_trip_durations)} total vehicle trips across {N_EVALUATION_EPISODES} episodes.")
        
        # Display key statistics for trip duration
        print("\n--- Trip Duration Stats ---")
        print(df['duration'].describe(percentiles=[.5, .75, .95]))

        # Display key statistics for waiting time
        print("\n--- Waiting Time Stats ---")
        print(df['waitingTime'].describe(percentiles=[.5, .75, .95]))

        # --- Visualization of the distribution ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df['duration'].plot(kind='hist', bins=50, title="Distribution of Trip Durations")
        plt.xlabel("Trip Duration (s)")

        plt.subplot(1, 2, 2)
        df['waitingTime'].plot(kind='hist', bins=50, title="Distribution of Waiting Times")
        plt.xlabel("Waiting Time (s)")
        
        plt.tight_layout()
        plt.show()