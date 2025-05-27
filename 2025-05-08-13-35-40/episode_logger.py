import csv
import os

class EpisodeLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        self.headers = ["episode", "worker", "steps", "avg_wait", "total_wait", "vehicles", "avg_speed", "trip_duration"]

        # Write header only once
        if not os.path.exists(self.output_path):
            with open(self.output_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, episode_num, worker_id, stats):
        row = [
            episode_num,
            worker_id,
            stats.get("steps", 0),
            round(stats.get("avg_wait", 0), 2),
            round(stats.get("avg_wait", 0) * stats.get("steps", 0), 2),
            stats.get("vehicles", 0),
            round(stats.get("avg_speed", 0), 2),
            round(stats.get("trip_duration", 0), 2)
        ]

        with open(self.output_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
