import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import subprocess
import traci
from sumolib import net


class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg, gui=False):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.step_count = 0
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"

        self.min_green_duration = 10
        self.max_green_duration = 60

        traci.start([self.sumo_binary, "-c", self.sumo_cfg])

        self.tls_ids = traci.trafficlight.getIDList()
        self.controlled_edges_map = {}
        self.phase_counts = []
        self.last_phase_change_step = {tls_id: -9999 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}
        self.active_green_duration = {tls_id: self.min_green_duration for tls_id in self.tls_ids}

        self.valid_phases_map = {
            "388514073": [0, 1],
            "485107212": [0, 1],
            "618601110": [0, 1],
            "618601236": [0, 1],
            "cluster_3594889293_485106994_7596245720": [0, 1],
            "cluster_3597298997_3597299013": [0, 1],
            "cluster_485108695_618976796": [0, 1],
            "cluster_549749941_618977140": [0, 1],
            "cluster_618601163_618601172": [0, 1]
        }
        self.phase_to_yellow_map = {
            "388514073": {0: 1, 2: 3},
            "485107212": {0: 1, 2: 3},
            "618601110": {0: 1, 2: 3},
            "618601236": {0: 1, 2: 3},
            "cluster_3594889293_485106994_7596245720": {0: 1, 2: 3},
            "cluster_3597298997_3597299013": {0: 1, 2: 3},
            "cluster_485108695_618976796": {0: 1, 2: 3},
            "cluster_549749941_618977140": {0: 1, 2: 3},
            "cluster_618601163_618601172": {0: 1, 2: 3}
        }

        for tls_id in self.tls_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_edges = list(set([lane.split("_")[0] for lane in controlled_lanes]))
            self.controlled_edges_map[tls_id] = controlled_edges

            num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
            self.phase_counts.append(num_phases)

        net_path = os.path.join(os.path.dirname(self.sumo_cfg), "osm.net.xml")
        self.net = net.readNet(net_path)
        self.all_observed_edges = []
        for tls_id in self.tls_ids:
            junction = self.net.getNode(tls_id)
            incoming_edges = [edge.getID() for edge in junction.getIncoming()]
            self.all_observed_edges.extend(e for e in incoming_edges if not e.startswith(":"))
        self.all_observed_edges = list(set(self.all_observed_edges))
            
        traci.close()

        self.action_space = spaces.Dict({
            tls_id: spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            for tls_id in self.tls_ids
        })

        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(len(self.all_observed_edges),), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        self.step_count = 0
        base_dir = os.path.dirname(__file__)

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

        traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        return self._get_observation(), {}


    def step(self, action):
        decoded_action = {}
        for i, tls_id in enumerate(self.tls_ids):
            phase_val = action[i][0] if isinstance(action[i], (list, np.ndarray)) else action[i]
            dur_val = action[i][1] if isinstance(action[i], (list, np.ndarray)) else 0.5
            decoded_action[tls_id] = [phase_val, dur_val]

        for tls_id in self.tls_ids:
            current_phase = traci.trafficlight.getPhase(tls_id)
            current_step = self.step_count
            time_since_last = current_step - self.last_phase_change_step[tls_id]

            if current_phase in self.valid_phases_map[tls_id]:
                if time_since_last >= self.active_green_duration[tls_id]:
                    yellow_phase = self.phase_to_yellow_map[tls_id][current_phase]
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    self.last_phase_change_step[tls_id] = current_step

            elif current_phase in self.phase_to_yellow_map[tls_id].values():
                green_choices = self.valid_phases_map[tls_id]
                phase_val, dur_val = decoded_action[tls_id]
                phase_index = int(phase_val * len(green_choices)) % len(green_choices)
                next_green = green_choices[phase_index]

                if next_green == self.last_actions[tls_id]:
                    phase_index = (phase_index + 1) % len(green_choices)
                    next_green = green_choices[phase_index]

                duration = int(self.min_green_duration + dur_val * (self.max_green_duration - self.min_green_duration))
                self.active_green_duration[tls_id] = duration

                traci.trafficlight.setPhase(tls_id, next_green)
                self.last_phase_change_step[tls_id] = current_step
                self.last_actions[tls_id] = next_green

        traci.simulationStep()
        self.step_count += 1

        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {}

        print(f"Step: {self.step_count}, Reward: {reward:.2f}, Done: {terminated}")
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        return np.array([
            traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges
        ], dtype=np.float32)

    def _get_reward(self):
        halted = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges)

        braking_penalty = 0
        slow_penalty = 0
        speed_sum = 0
        vehicle_count = 0

        for v in traci.vehicle.getIDList():
            accel = traci.vehicle.getAcceleration(v)
            speed = traci.vehicle.getSpeed(v)

            if accel < -3.0:
                braking_penalty += 1
            if speed < 2.0:
                slow_penalty += 1

            speed_sum += speed
            vehicle_count += 1

        avg_speed = speed_sum / vehicle_count if vehicle_count > 0 else 0
        avg_halted = halted / vehicle_count if vehicle_count > 0 else 0
        avg_slow = slow_penalty / vehicle_count if vehicle_count > 0 else 0

        arrived = traci.simulation.getArrivedNumber()
        teleports = traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber()
       
        reward = (
            + 0.5 * arrived
            + 0.5 * avg_speed
            - 2.0 * avg_halted
            - 1.5 * avg_slow
            - 2.0 * teleports
        )

        return reward

    def _is_done(self):
        MAX_STEPS = 2000
        return traci.simulation.getMinExpectedNumber() == 0 or self.step_count >= MAX_STEPS

    def render(self):
        pass

    def close(self):
        if traci.isLoaded():
            traci.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

