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
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"

        # --- Environment Parameters ---
        # Define the discrete options for green light durations in seconds.
        self.duration_options = [15, 30, 45, 60]
        # Maximum plausible green time for normalization purposes
        self.max_green_time_for_norm = 120.0

        # --- Initialize State Variables ---
        self.step_count = 0
        self.last_arrived_count = 0
        self.last_teleport_count = 0

        # --- Connect to SUMO to get network info ---
        traci.start([self.sumo_binary, "-c", self.sumo_cfg])

        self.tls_ids = traci.trafficlight.getIDList()
        
        self.valid_phases_map = {
            "2315558178": [0, 2], "2622084397": [0, 2], "388514073": [0, 2],
            "485107212": [0, 2], "618601110": [0, 2], "618601236": [0, 2],
            "618601338": [0, 2, 4], "6289298632": [0, 2], "6291986227": [0, 2, 4],
            "J15": [0, 2, 4, 6], "J6": [0, 2],
            "cluster_3594889293_485106994_7596245720": [0, 2],
            "cluster_3597298997_3597299013": [0, 2],
            "cluster_485108695_618976796": [0, 2],
            "cluster_549749941_618977140": [0, 2],
            "cluster_618601163_618601172": [0, 2],
        }

        self.phase_to_yellow_map = {
            "2315558178": {0: 1, 2: 3}, "2622084397": {0: 1, 2: 3},
            "388514073": {0: 1, 2: 3}, "485107212": {0: 1, 2: 3},
            "618601110": {0: 1, 2: 3}, "618601236": {0: 1, 2: 3},
            "618601338": {0: 1, 2: 3, 4: 5}, "6289298632": {0: 1, 2: 3},
            "6291986227": {0: 1, 2: 3, 4: 5}, "J15": {0: 1, 2: 3, 4: 5, 6: 7},
            "J6": {0: 1, 2: 3},
            "cluster_3594889293_485106994_7596245720": {0: 1, 2: 3},
            "cluster_3597298997_3597299013": {0: 1, 2: 3},
            "cluster_485108695_618976796": {0: 1, 2: 3},
            "cluster_549749941_618977140": {0: 1, 2: 3},
            "cluster_618601163_618601172": {0: 1, 2: 3},
        }
        
        net_path = os.path.join(os.path.dirname(self.sumo_cfg), "osm.net.xml")
        self.net = net.readNet(net_path)
        self.all_observed_edges = []
        for tls_id in self.tls_ids:
            junction = self.net.getNode(tls_id)
            # Collect all edges connected to this traffic light
            for edge in junction.getOutgoing() + junction.getIncoming():
                if not edge.getID().startswith(':'):
                    self.all_observed_edges.append(edge.getID())
        # Remove duplicates and sort the edges
        self.all_observed_edges = sorted(list(set(self.all_observed_edges)))
            
        traci.close()

        # --- Initialize Per-Episode State Attributes ---
        self.last_phase_change_step = {}
        self.last_actions = {}
        self.active_green_duration = {}

        # --- This is a fail-safe for a specific intersection, which has a lazy learning pathern
        #    where it gets stuck on letting pedestrians cross without ever changing the phase.

        self.problematic_tls_id = "388514073"
        self.maybe_problematic_tls_ids = "6291986227"
        self.max_red_time = 120

        # --- Define Action and Observation Spaces ---
        # NEW ACTION SPACE: Use MultiDiscrete for phase and duration choices.
        self.action_space = spaces.Dict({
            tls_id: spaces.MultiDiscrete([ len(self.valid_phases_map[tls_id]), len(self.duration_options) ])
            for tls_id in self.tls_ids
        })

        # NEW OBSERVATION SPACE: Add a dimension for the elapsed time of each traffic light.
        num_edge_observations = len(self.all_observed_edges) * 3 # Halting, total, mean
        num_tls_observations = len(self.tls_ids) * 2 # Current phase, elapsed time in current phase

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_edge_observations + num_tls_observations,),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        base_dir = os.path.dirname(__file__)

        # Reset the environment state for a new episode
        self.step_count = 0
        self.last_arrived_count = 0
        self.last_teleport_count = 0
        self.last_phase_change_step = {tls_id: 0 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}
        # Start with a default duration for all lights
        self.active_green_duration = {tls_id: self.duration_options[0] for tls_id in self.tls_ids}
        

           # Commented out the subprocess calls for generating trips and routes in idea of using pre-generated files for training.


        subprocess.run(["python", os.path.join(base_dir, "simulation_generator.py"),
                        os.path.join(base_dir, "full.rou.xml")], check=True)
        subprocess.run(["python", os.path.join(base_dir, "truck_generator.py")], check=True)

        duarouter_cmd = [
            "duarouter",
            "-n", os.path.join(base_dir, "osm.net.xml"),
            "--trip-files", os.path.join(base_dir, "trips.pedestrians.xml"),
            "-a", os.path.join(base_dir, "vtypes.xml"),
            "-o", os.path.join(base_dir, "ped.rou.xml")
        ]

        print("[INFO] Running duarouter for vehicle + pedestrian trips...")
        result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] duarouter failed:")
            print(result.stderr)





        traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        return self._get_observation(), {}


    def step(self, action):

        action_dict = {}
        # Made to handle evaluation process
        if isinstance(action, dict):
            action_dict = action
        else:
            action_idx = 0
            for tls_id in self.tls_ids:
                action_dict[tls_id] = [action[action_idx], action[action_idx + 1]]
                action_idx += 2

        # --- Now, decode directly from the standardized action_dict ---
        decoded_action = {}
        for tls_id, act_indices in action_dict.items():
            phase_choice_idx = act_indices[0]
            duration_choice_idx = act_indices[1]

            # Look up the actual phase and duration from our predefined options
            next_phase = self.valid_phases_map[tls_id][phase_choice_idx]
            next_duration = self.duration_options[duration_choice_idx]

            decoded_action[tls_id] = {"phase": next_phase, "duration": next_duration}

        # 2. Apply the actions to the simulation logic
        for tls_id in self.tls_ids:
            current_phase = traci.trafficlight.getPhase(tls_id)
            time_since_last = self.step_count - self.last_phase_change_step[tls_id]

            # The actual fail-safe for the problematic traffic light
            if tls_id == self.problematic_tls_id:
                # Check if current phase is for pedestrians
                if current_phase ==2 or current_phase == 3:
                    def_time_on_red = self.step_count - self.last_phase_change_step.get(tls_id, 0)
                    if def_time_on_red >= self.max_red_time:
                        print(f"FAILSAFE: Forcing TLS {tls_id} to green for vehicles at step {self.step_count}.")
                        # Force the phase to GREEN for cars (phase 0)
                        traci.trafficlight.setPhase(tls_id, 0) 
                        # Force the default green duration for this override
                        self.active_green_duration[tls_id] = self.duration_options[3] # e.g., 60 seconds
                        # Reset the timer for this new phase
                        self.last_phase_change_step[tls_id] = self.step_count
                        continue # Skip the agent's logic for this light this step
            
            # Handle the maybe problematic TLS ID
            if tls_id == self.maybe_problematic_tls_ids:
                if current_phase == 4 or current_phase == 5:
                    maybe_time_on_red = self.step_count - self.last_phase_change_step.get(tls_id, 0)
                    if maybe_time_on_red >= self.max_red_time:
                        print(f"FAILSAFE: Forcing TLS {tls_id} to green for vehicles at step {self.step_count}.")
                        # Force the phase to GREEN for cars (phase 2), phase with least cars
                        traci.trafficlight.setPhase(tls_id, 2) 
                        # Force the default green duration for this override
                        self.active_green_duration[tls_id] = self.duration_options[0] # e.g., 15 seconds
                        # Reset the timer for this new phase
                        self.last_phase_change_step[tls_id] = self.step_count
                        continue # Skip the agent's logic for this light this step
                    

            if current_phase in self.valid_phases_map[tls_id]:
                if time_since_last >= self.active_green_duration[tls_id]:
                    yellow_phase = self.phase_to_yellow_map[tls_id][current_phase]
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    self.last_phase_change_step[tls_id] = self.step_count

            elif current_phase in self.phase_to_yellow_map[tls_id].values():
                yellow_duration = 3

                # Check if the yellow light has been on for its full duration
                if time_since_last >= yellow_duration:
                    next_green_phase = decoded_action[tls_id]["phase"]
                    new_green_duration = decoded_action[tls_id]["duration"]

                    traci.trafficlight.setPhase(tls_id, next_green_phase)

                    self.active_green_duration[tls_id] = new_green_duration
                    self.last_phase_change_step[tls_id] = self.step_count
                    self.last_actions[tls_id] = next_green_phase

        # 3. Step the simulation and get results
        traci.simulationStep()
        self.step_count += 1     
        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {}
        
        if terminated:
            vehicle_types = ["pedestrian", "passenger", "truck", "bicycle", "motorcycle", "delivery", "emergency"]
            total_wait_times = {v_type: 0.0 for v_type in vehicle_types}
            vehicle_counts = {v_type: 0 for v_type in vehicle_types}
            try:
                for veh_id in traci.vehicle.getIDList():
                    v_type = traci.vehicle.getTypeID(veh_id)
                    if v_type in vehicle_types:
                        wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                        total_wait_times[v_type] += wait_time
                        vehicle_counts[v_type] += 1
            except traci.TraCIException:
                pass # Ignore if simulation closes during this loop
            for v_type in vehicle_types:
                if vehicle_counts[v_type] > 0:
                    info[f'wait_time/{v_type}'] = total_wait_times[v_type] / vehicle_counts[v_type]
                else:
                    info[f'wait_time/{v_type}'] = 0.0

        # Printing debug information
        # print(f"Step: {self.step_count}, Reward: {reward:.2f}, Done: {terminated}")
        
        return obs, reward, terminated, truncated, info
    
    # def _get_observation(self):
    #     # --- Halting vehicles, normalized ---
    #     halting_vehicles = [
    #         traci.edge.getLastStepHaltingNumber(e) / 200.0 for e in self.all_observed_edges
    #     ]
    #     halting_vehicles = np.clip(halting_vehicles, 0, 1)

    #     # --- Current phase index, normalized ---
    #     current_phases = [
    #         traci.trafficlight.getPhase(tls_id) / 10.0 for tls_id in self.tls_ids
    #     ]
        
    #     # --- NEW: Time elapsed in current phase, normalized ---
    #     elapsed_times = [
    #         (self.step_count - self.last_phase_change_step[tls_id]) / self.max_green_time_for_norm
    #         for tls_id in self.tls_ids
    #     ]
    #     elapsed_times = np.clip(elapsed_times, 0, 1)
        
    #     # --- Combine all observations into a single flat array ---
    #     obs = np.concatenate([halting_vehicles, current_phases, elapsed_times]).astype(np.float32)
    #     return obs

    def _get_observation(self):
        # --- Edge-based Observations (Normalized) ---
        halting_counts = []
        vehicle_counts = []
        mean_speeds = []
        
        # Constants for normalization
        MAX_VEHICLES_PER_EDGE = 200.0
        MAX_SPEED = 13.89 # Approx 50 km/h

        for edge in self.all_observed_edges:
            halting_counts.append(traci.edge.getLastStepHaltingNumber(edge) / MAX_VEHICLES_PER_EDGE)
            vehicle_counts.append(traci.edge.getLastStepVehicleNumber(edge) / MAX_VEHICLES_PER_EDGE)
            # getLastStepMeanSpeed returns speed in m/s
            mean_speeds.append(traci.edge.getLastStepMeanSpeed(edge) / MAX_SPEED)

        # --- TLS-based Observations (Normalized) ---
        current_phases = [traci.trafficlight.getPhase(tls_id) / 10.0 for tls_id in self.tls_ids]
        elapsed_times = [(self.step_count - self.last_phase_change_step[tls_id]) / self.max_green_time_for_norm for tls_id in self.tls_ids]

        # --- Combine all observations into a single flat array ---
        obs = np.concatenate([
            np.clip(halting_counts, 0, 1),
            np.clip(vehicle_counts, 0, 1),
            np.clip(mean_speeds, 0, 1),
            np.clip(current_phases, 0, 1),
            np.clip(elapsed_times, 0, 1)
        ]).astype(np.float32)
        
        return obs
        

    def _get_reward(self):
        # --- 1. Penalty for Congestion ---
        total_halting_vehicles = sum(
            traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges
        )
        # Make the penalty for each waiting car slightly higher
        congestion_penalty = -total_halting_vehicles * 1.5

        # --- 2. Reward for Throughput ---
        current_arrived_count = traci.simulation.getArrivedNumber()
        newly_arrived = current_arrived_count - self.last_arrived_count
        self.last_arrived_count = current_arrived_count
        # Reduce the bonus to discourage "farming" this reward
        throughput_reward = newly_arrived * 5.0 

        # --- 3. Penalty for Gridlock (Keep this high) ---
        current_teleport_count = traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber()
        newly_teleported = current_teleport_count - self.last_teleport_count
        self.last_teleport_count = current_teleport_count
        gridlock_penalty = -newly_teleported * 500.0

        # --- 4. Add a penalty for every step to encourage speed ---
        time_penalty = -0.5

        # --- 5. Add a penalty for hard breaking ---
        breaking_penalty = 0
        for veh_id in traci.vehicle.getIDList():
            #Check for the accelerations less than a harsh breaking threshold
            if traci.vehicle.getAcceleration(veh_id) < -4.5:
                breaking_penalty -= 1.0
        
        # --- 6. Add a penalty for vehicles waiting at red lights ---
        all_wait_times = [traci.vehicle.getAccumulatedWaitingTime(v_id) for v_id in traci.vehicle.getIDList()]
        max_wait_time = max(all_wait_times) if all_wait_times else 0
        # Use a quadratic penalty for waiting time to discourage excessive delays
        max_time_penalty = -(max_wait_time ** 2) * 0.01

        # --- 7. Stale Phase Penalty ---
        stale_phase_penalty = 0
        for tls_id in self.tls_ids:
            time_in_current_phase = self.step_count - self.last_phase_change_step.get(tls_id, self.step_count)
            # Penalty starts low and increases with time in the same phase
            stale_phase_penalty -= time_in_current_phase * 0.025

        # --- 7. Combine and Scale the Reward ---
        combined_reward = congestion_penalty + throughput_reward + gridlock_penalty + time_penalty + breaking_penalty + max_time_penalty + stale_phase_penalty
        scaled_reward = combined_reward / 1000.0

        return scaled_reward

    def _is_done(self):
        MAX_STEPS = 3600
        return traci.simulation.getMinExpectedNumber() == 0 or self.step_count >= MAX_STEPS

    def render(self):
        pass

    def close(self):
        if traci.isLoaded():
            traci.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]