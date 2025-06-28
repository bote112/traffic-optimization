import os
import subprocess
import traci
import sumolib
import statistics
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

# Paths and config
USE_GUI = False
CONFIG_FILE = "osm.sumocfg"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parameters
EPISODES = 1000
STEPS = 2000
SUMO_BINARY = "sumo-gui" if USE_GUI else "sumo"

# TensorBoard Logging
TENSORBOARD_LOGDIR = os.path.join(BASE_DIR, "tensorboard_baseline")
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

# Storage for metrics
episode_travel_times = []
episode_waiting_times = []
episode_vehicle_counts = []

def run_episode(episode_num):
    print(f"Running episode {episode_num + 1}")

    # Generate new traffic scenario
    subprocess.run(["python", "simulation_generator.py", "full.rou.xml"], check=True)
    subprocess.run(["python", "truck_generator.py"], check=True)

    # Route vehicle + pedestrian trips using duarouter
    duarouter_cmd = [
        "duarouter",
        "-n", os.path.join(BASE_DIR, "osm.net.xml"),
        "--trip-files", os.path.join(BASE_DIR, "trips.pedestrians.xml"),
        "-a", os.path.join(BASE_DIR, "vtypes.xml"),
        "-o", os.path.join(BASE_DIR, "ped.rou.xml")
    ]

    result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] duarouter failed:")
        print(result.stderr)
        return 0, 0, 0

    # Start SUMO with tripinfo output
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", "tripinfo.xml"])
    
    step = 0
    while step < STEPS and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

    traci.close()

    # Parse tripinfo for metrics
    tree = ET.parse("tripinfo.xml")
    root = tree.getroot()

    travel_times = []
    waiting_times = []

    for trip in root.findall('tripinfo'):
        travel_times.append(float(trip.attrib['duration']))
        waiting_times.append(float(trip.attrib['waitingTime']))

    avg_travel = statistics.mean(travel_times) if travel_times else 0
    avg_waiting = statistics.mean(waiting_times) if waiting_times else 0
    vehicle_count = len(travel_times)

    return avg_travel, avg_waiting, vehicle_count

if __name__ == "__main__":
    for i in range(EPISODES):
        avg_travel, avg_waiting, count = run_episode(i)

        episode_travel_times.append(avg_travel)
        episode_waiting_times.append(avg_waiting)
        episode_vehicle_counts.append(count)

        # TensorBoard logging
        writer.add_scalar("baseline/avg_travel_time", avg_travel, i)
        writer.add_scalar("baseline/avg_waiting_time", avg_waiting, i)
        writer.add_scalar("baseline/vehicle_count", count, i)

    writer.close()

    # Compute overall averages
    avg_travel_all = statistics.mean(episode_travel_times)
    avg_waiting_all = statistics.mean(episode_waiting_times)
    avg_vehicle_count = statistics.mean(episode_vehicle_counts)

    print("\nBaseline Summary:")
    print(f"Avg Travel Time: {avg_travel_all:.2f} s")
    print(f"Avg Waiting Time: {avg_waiting_all:.2f} s")
    print(f"Avg Vehicles per Episode: {avg_vehicle_count:.1f}")

    # --- Visualization ---
    episodes = list(range(1, EPISODES + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_travel_times, marker='o', label='Average Travel Time')
    plt.plot(episodes, episode_waiting_times, marker='x', label='Average Waiting Time')
    plt.xlabel("Episode")
    plt.ylabel("Time (seconds)")
    plt.title("Baseline Traffic Performance Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
