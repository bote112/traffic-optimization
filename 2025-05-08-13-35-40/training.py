import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from parallel_runner import run_worker
from episode_logger import EpisodeLogger

# === Config ===
NUM_EPISODES = 1500
BATCH_SIZE = 5
assert NUM_EPISODES % BATCH_SIZE == 0, "NUM_EPISODES must be divisible by BATCH_SIZE"

# === Results ===
all_waiting_times = []
all_episode_lengths = []
all_total_halted = []

def run_parallel_episodes(num_workers=8):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(target=run_worker, args=(i, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return dict(return_dict)

# === MAIN RUNNER ===
if __name__ == '__main__':

    output_csv = os.path.join("E:/licenta/oras-mic/2025-05-08-13-35-40", "results.csv")
    logger = EpisodeLogger(output_csv)
    if not os.path.exists(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv))




    num_batches = NUM_EPISODES // BATCH_SIZE
    for batch in range(num_batches):
        print(f"\n[Batch {batch + 1}/{num_batches}] Running {BATCH_SIZE} episodes...")
        results = run_parallel_episodes(num_workers=BATCH_SIZE)

        episode_base = batch * BATCH_SIZE
        for wid, res in results.items():
            if "error" in res:
                print(f"[Worker {wid}] Error: {res['error']}")
            else:
                steps = res["steps"]
                avg_wait = res["avg_wait"]
                total_halted = avg_wait * steps
                vehicles = res["vehicles"]
                avg_speed = res["avg_speed"]
                trip_duration = res["trip_duration"]
                episode_num = episode_base + wid + 1
                logger.log(episode_num, wid, res)
                print(f"Logging episode data to: {logger.output_path}")



                all_waiting_times.append(avg_wait)
                all_episode_lengths.append(steps)
                all_total_halted.append(total_halted)

                print(f"[Worker {wid}] Steps: {steps}, Avg Wait: {avg_wait:.2f}, Total Halted: {total_halted:.0f}")

    # === Plotting ===
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
