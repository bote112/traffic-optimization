# simulation_generator.py
import os
import random
import subprocess
import sys

# === CONFIG ===
SUMO_HOME = os.environ.get("SUMO_HOME", "C:/Program Files (x86)/Eclipse/Sumo")
TOOLS_PATH = os.path.join(SUMO_HOME, "tools")
BASE_PATH = "E:/licenta/oras-mic/2025-05-08-13-35-40"
NETWORK_FILE = os.path.join(BASE_PATH, "osm.net.xml")
TRIPS_FILE = os.path.join(BASE_PATH, "trips.trips.xml")
TRIP_DURATION = 600  # Shorter for faster simulation

# === Dynamic Route Output ===
if len(sys.argv) < 2:
    #print("Usage: python simulation_generator.py <output_route_file>")
    sys.exit(1)

ROUTES_FILE = sys.argv[1]

# === RANDOMIZED PARAMETERS ===
period = round(random.uniform(1.60, 2.5), 2)
seed = random.randint(0, 99999)
fringe = round(random.uniform(1.0, 4.0), 1)

#print(f"[GENERATOR] Creating route: {ROUTES_FILE}")
#print(f"[GENERATOR] Parameters -> seed: {seed}, period: {period}, fringe: {fringe}")


# === Build command ===
command = [
    "python", os.path.join(TOOLS_PATH, "randomTrips.py"),
    "-n", NETWORK_FILE,
    "-o", TRIPS_FILE,
    "-r", ROUTES_FILE,
    "--seed", str(seed),
    "--period", str(period),
    "--fringe-factor", str(fringe),
    "--prefix", "veh",
    "--trip-attributes", 'departLane="best" departSpeed="0" departPos="random"',
    "--end", str(TRIP_DURATION)
]

result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print()
    #print(f"[INFO] Routes generated at {ROUTES_FILE}")
else:
    print("[ERROR] Route generation failed:")
    print(result.stderr)
    sys.exit(1)
