import os
import subprocess
import uuid
import traci
import multiprocessing
import sys

# === Paths ===
BASE_PATH = "E:/licenta/oras-mic/2025-05-08-13-35-40"
ORIGINAL_CFG = os.path.join(BASE_PATH, "osm.sumocfg")
NETWORK_FILE = os.path.join(BASE_PATH, "osm.net.xml")
POLY_FILE = os.path.join(BASE_PATH, "osm.poly.xml")
SIM_GEN = os.path.join(BASE_PATH, "simulation_generator.py")

ROUTE_DIR = os.path.join(BASE_PATH, "generated_routes")
os.makedirs(ROUTE_DIR, exist_ok=True)

# Optional: suppress traci.start() output
import contextlib
import sys

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def run_worker(worker_id, return_dict):
    try:
        unique_id = uuid.uuid4().hex[:8]
        route_file = os.path.join(ROUTE_DIR, f"routes_worker{worker_id}_{unique_id}.rou.xml")
        temp_cfg = os.path.join(ROUTE_DIR, f"temp_worker{worker_id}_{unique_id}.sumocfg")

        # Generate route file with suppressed output
        subprocess.run(["python", SIM_GEN, route_file], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Modify config with proper paths
        with open(ORIGINAL_CFG, 'r') as f:
            cfg_content = f.read()
            cfg_content = cfg_content.replace("routes.rou.xml", route_file)
            cfg_content = cfg_content.replace("osm.net.xml", NETWORK_FILE.replace("\\", "/"))
            cfg_content = cfg_content.replace("osm.poly.xml", POLY_FILE.replace("\\", "/"))

        with open(temp_cfg, 'w') as f:
            f.write(cfg_content)

        sumo_cmd = [
            "sumo",
            "-c", temp_cfg,
            "--step-length", "0.1",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--no-warnings", "true",
            "--verbose", "false"
        ]

        # Start SUMO silently
        with suppress_output():
            traci.start(sumo_cmd)

        # Run simulation
        steps = 0
        total_wait = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            edges = traci.edge.getIDList()
            halted = sum(traci.edge.getLastStepHaltingNumber(e) for e in edges if not e.startswith(":"))
            total_wait += halted
            steps += 1

        # Collect stats
        vehicles = traci.simulation.getArrivedNumber()
        avg_speed = sum(traci.edge.getLastStepMeanSpeed(e) for e in edges if not e.startswith(":")) / len(edges)
        trip_duration = 0  

        traci.close()

        avg_wait = total_wait / steps if steps > 0 else 0

        return_dict[worker_id] = {
            "steps": steps,
            "avg_wait": avg_wait,
            "vehicles": vehicles,
            "avg_speed": avg_speed,
            "trip_duration": trip_duration
        }

    except Exception as e:
        if traci.isLoaded():
            traci.close()
        return_dict[worker_id] = {"error": str(e)}
