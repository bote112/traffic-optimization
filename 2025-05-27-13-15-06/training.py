import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import traci
from episode_logger import EpisodeLogger
from torch.utils.tensorboard import SummaryWriter  # <-- TensorBoard

# === CONFIG ===
NUM_EPISODES = 1500
SUMO_CFG = "osm.sumocfg"
USE_GUI = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Paths ===
output_dir = os.path.join(BASE_DIR, "baseline_results")
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "results.csv")
logger = EpisodeLogger(output_csv)

simulation_generator = os.path.join(BASE_DIR, "simulation_generator.py")
truck_generator = os.path.join(BASE_DIR, "truck_generator.py")

sumo_binary = "sumo-gui" if USE_GUI else "sumo"

# === TensorBoard Writer ===
TENSORBOARD_LOGDIR = os.path.join(BASE_DIR, "baseline_tb_logs")
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

# === DATA ===
all_waiting_times = []
all_episode_lengths = []
all_total_halted = []

# === MAIN LOOP ===
for episode in range(1, NUM_EPISODES + 1):
    print(f"\n[Episode {episode}/{NUM_EPISODES}] Generating simulation data...")

    try:
        # Run trip and route generation
        subprocess.run(["python", simulation_generator, "full.rou.xml"], check=True)
        subprocess.run(["python", truck_generator], check=True)

        # Route vehicle + pedestrian trips using duarouter
        duarouter_cmd = [
            "duarouter",
            "-n", os.path.join(BASE_DIR, "osm.net.xml"),
            "--trip-files", os.path.join(BASE_DIR, "trips.pedestrians.xml"),
            "-a", os.path.join(BASE_DIR, "vtypes.xml"),
            "-o", os.path.join(BASE_DIR, "ped.rou.xml")
        ]

        print("[INFO] Running duarouter for vehicle + pedestrian trips...")
        result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] duarouter failed:")
            print(result.stderr)
            continue

        print("[INFO] Starting SUMO simulation...")
        traci.start([sumo_binary, "-c", SUMO_CFG, "--step-length", "1"])
        total_wait = 0
        steps = 0
        vehicle_count = 0
        speed_total = 0

        while traci.simulation.getMinExpectedNumber() > 0 and steps < 5000:
            traci.simulationStep()
            edges = traci.edge.getIDList()
            for edge in edges:
                if not edge.startswith(":"):
                    total_wait += traci.edge.getLastStepHaltingNumber(edge)
                    speed_total += traci.edge.getLastStepMeanSpeed(edge)
                    vehicle_count += 1
            steps += 1

        avg_wait = total_wait / steps if steps else 0
        avg_speed = speed_total / vehicle_count if vehicle_count else 0
        total_halted = avg_wait * steps

        result = {
            "steps": steps,
            "avg_wait": avg_wait,
            "vehicles": vehicle_count,
            "avg_speed": avg_speed,
            "trip_duration": 0  # optional
        }

        logger.log(episode, 0, result)

        # TensorBoard Logging
        writer.add_scalar("baseline/avg_wait", avg_wait, episode)
        writer.add_scalar("baseline/avg_speed", avg_speed, episode)
        writer.add_scalar("baseline/vehicles", vehicle_count, episode)
        writer.add_scalar("baseline/episode_length", steps, episode)

        all_waiting_times.append(avg_wait)
        all_episode_lengths.append(steps)
        all_total_halted.append(total_halted)

        np.savez("autosave_data.npz",
                 waiting_times=np.array(all_waiting_times),
                 episode_lengths=np.array(all_episode_lengths),
                 total_halted=np.array(all_total_halted))

        print(f"[Episode {episode}] Steps: {steps}, Avg Wait: {avg_wait:.2f}, Halted: {total_halted:.0f}")

    except Exception as e:
        print(f"[ERROR] Episode {episode} failed: {str(e)}")

    finally:
        if traci.isLoaded():
            traci.close()

# === Cleanup ===
writer.close()

# === PLOT ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(all_waiting_times)
plt.title("Avg Waiting Time per Episode")
plt.xlabel("Episode")
plt.ylabel("Avg Waiting Time")

plt.subplot(1, 3, 2)
plt.plot(all_total_halted)
plt.title("Total Halted Vehicles per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Halted")

plt.subplot(1, 3, 3)
plt.plot(all_episode_lengths)
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Episode Length")

plt.tight_layout()
plt.show()
