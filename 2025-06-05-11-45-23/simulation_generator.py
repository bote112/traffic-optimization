import os
import random
import subprocess
import sys

# === CONFIG ===
SUMO_HOME = os.environ.get("SUMO_HOME", "C:/Program Files (x86)/Eclipse/Sumo")
TOOLS_PATH = os.path.join(SUMO_HOME, "tools")
BASE_PATH = "E:/licenta/oras-mediu/2025-06-05-11-45-23"

NETWORK_FILE = os.path.join(BASE_PATH, "osm.net.xml")
TRIPS_FILE_VEH = os.path.join(BASE_PATH, "trips.vehicles.xml")
TRIPS_FILE_PED = os.path.join(BASE_PATH, "trips.pedestrians.xml")
ROUTES_FILE = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Usage: python simulation_generator.py <output_routes.rou.xml>")

TRIP_DURATION = 1200  # Simulation duration in seconds

# === RANDOM PARAMETERS ===
period = round(random.uniform(0.4, 0.75), 2)
seed = random.randint(0, 99999)
fringe = round(random.uniform(1.0, 4.0), 1)

print(f"[INFO] Seed: {seed} | Period: {period} | Fringe: {fringe}")

# === Generate VEHICLE trips using only allowed vTypes (no truck, no bus) ===
command_veh = [
    "python", os.path.join(TOOLS_PATH, "randomTrips.py"),
    "-n", NETWORK_FILE,
    "-o", TRIPS_FILE_VEH,
    "-r", ROUTES_FILE,
    "--seed", str(seed),
    "--period", str(period),
    "--fringe-factor", str(fringe),
    "--prefix", "veh",
    "--trip-attributes", 'departLane="best" departSpeed="0" departPos="random" type="randomDist"',
    "--end", str(TRIP_DURATION)
]

print("[INFO] Generating non-truck vehicle trips...")
result_veh = subprocess.run(command_veh, capture_output=True, text=True)
if result_veh.returncode == 0:
    print(f"[INFO] Vehicle trips generated: {TRIPS_FILE_VEH}")
else:
    print("[ERROR] Vehicle trip generation failed:")
    print(result_veh.stderr)
    sys.exit(1)

# === Generate PEDESTRIAN trips ===
command_ped = [
    "python", os.path.join(TOOLS_PATH, "randomTrips.py"),
    "-n", NETWORK_FILE,
    "-o", TRIPS_FILE_PED,
    "--prefix", "ped",
    "--pedestrian",
    "--end", str(TRIP_DURATION),
    "--seed", str(seed + 1),
    "--period", "5.0",
    "--max-distance", "500",
    "--validate"
]

print("[INFO] Generating pedestrian trips...")
result_ped = subprocess.run(command_ped, capture_output=True, text=True)
if result_ped.returncode == 0:
    print(f"[INFO] Pedestrian trips generated: {TRIPS_FILE_PED}")
else:
    print("[ERROR] Pedestrian trip generation failed:")
    print(result_ped.stderr)
    sys.exit(1)
