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
            "2315558178": [0, 2],
            "2622084397": [0, 2],
            "388514073": [0, 2],
            "485107212": [0, 2],
            "618601110": [0, 2],
            "618601236": [0, 2],
            "618601338": [0, 2, 4],
            "6289298632": [0, 2],
            "6291986227": [0, 2, 4],
            "J15": [0, 2, 4, 6],
            "J6": [0, 2],
            "cluster_3594889293_485106994_7596245720": [0, 2],
            "cluster_3597298997_3597299013": [0, 2],
            "cluster_485108695_618976796": [0, 2],
            "cluster_549749941_618977140": [0, 2],
            "cluster_618601163_618601172": [0, 2],
        }

        self.phase_to_yellow_map = {
            "2315558178": {0: 1, 2: 3},
            "2622084397": {0: 1, 2: 3},
            "388514073": {0: 1, 2: 3},
            "485107212": {0: 1, 2: 3},
            "618601110": {0: 1, 2: 3},
            "618601236": {0: 1, 2: 3},
            "618601338": {0: 1, 2: 3, 4: 5},
            "6289298632": {0: 1, 2: 3},
            "6291986227": {0: 1, 2: 3, 4: 5},
            "J15": {0: 1, 2: 3, 4: 5, 6: 7},
            "J6": {0: 1, 2: 3},
            "cluster_3594889293_485106994_7596245720": {0: 1, 2: 3},
            "cluster_3597298997_3597299013": {0: 1, 2: 3},
            "cluster_485108695_618976796": {0: 1, 2: 3},
            "cluster_549749941_618977140": {0: 1, 2: 3},
            "cluster_618601163_618601172": {0: 1, 2: 3},
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

        self.last_arrived_count = 0
        self.last_teleport_count = 0

        self.action_space = spaces.Dict({
            tls_id: spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            for tls_id in self.tls_ids
        })

        num_tls = len(self.tls_ids) 
        num_edges = len(self.all_observed_edges)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_edges+num_tls,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        # Reset the environment state
        self.last_arrived_count = 0
        self.last_teleport_count = 0

        self.step_count = 0
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





        traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        return self._get_observation(), {}


    def step(self, action):
        # 'action' is a flat numpy array like: [tls1_act1, tls1_act2, tls2_act1, tls2_act2, ...]
        
        # We need to reconstruct the mapping from tls_id to its specific action
        decoded_action = {}
        action_idx = 0
        for tls_id in self.tls_ids:
            # Each action for a traffic light consists of 2 values (phase and duration)
            phase_val = action[action_idx]
            dur_val = action[action_idx + 1]
            decoded_action[tls_id] = [phase_val, dur_val]
            action_idx += 2 # Move the index for the next traffic light's action

        # The rest of your logic can now use this 'decoded_action' dictionary,
        # which correctly maps each traffic light to its chosen action.
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
                
                # Get the action for this specific tls_id from our decoded dictionary
                phase_val, dur_val = decoded_action[tls_id]
                
                phase_index = int(phase_val * len(green_choices)) % len(green_choices)
                next_green = green_choices[phase_index]

                # Optional but good: prevent choosing the same phase again if other options exist
                if len(green_choices) > 1 and next_green == self.last_actions[tls_id]:
                    phase_index = (phase_index + 1) % len(green_choices)
                    next_green = green_choices[phase_index]

                duration = int(self.min_green_duration + dur_val * (self.max_green_duration - self.min_green_duration))
                self.active_green_duration[tls_id] = duration

                traci.trafficlight.setPhase(tls_id, next_green)
                self.last_phase_change_step[tls_id] = current_step
                self.last_actions[tls_id] = next_green

        # --- Step the simulation and get results ---
        traci.simulationStep()
        self.step_count += 1

        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {}

         # When the episode is over, add the final stats you want to log
        if terminated:
            
            # 1. Define the vehicle types you want to track
            vehicle_types = ["pedestrian", "passenger", "truck", "bicycle", "motorcycle", "delivery", "emergency"]
            
            # 2. Initialize dictionaries to store total wait times and counts
            total_wait_times = {v_type: 0.0 for v_type in vehicle_types}
            vehicle_counts = {v_type: 0 for v_type in vehicle_types}

            # 3. Loop through all vehicles at the end of the simulation
            # Note: This includes vehicles that have already arrived
            for veh_id in traci.vehicle.getIDList():
                try:
                    v_type = traci.vehicle.getTypeID(veh_id)
                    if v_type in vehicle_types:
                        # Get total accumulated waiting time for this vehicle
                        wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                        total_wait_times[v_type] += wait_time
                        vehicle_counts[v_type] += 1
                except traci.TraCIException:
                    # Handle cases where a vehicle might disappear between getIDList and getTypeID
                    continue

            # 4. Calculate averages and add them to the info dictionary
            for v_type in vehicle_types:
                # IMPORTANT: Check for division by zero if a vehicle type was not in the episode
                if vehicle_counts[v_type] > 0:
                    avg_wait_time = total_wait_times[v_type] / vehicle_counts[v_type]
                    # The Monitor wrapper will pick up any key formatted like "wait_time/..."
                    info[f'wait_time/{v_type}'] = avg_wait_time
                else:
                    # If no vehicles of this type, log a waiting time of 0
                    info[f'wait_time/{v_type}'] = 0.0
                    

        # It's a good idea to remove this print statement during long training runs
        # as it will slow things down.
        print(f"Step: {self.step_count}, Reward: {reward:.2f}, Done: {terminated}")
        
        return obs, reward, terminated, truncated, info
    

    def _get_observation(self):

        MAX_HALTING_VEHICLES = 200

        halting_vehicles = [
            traci.edge.getLastStepHaltingNumber(e) / MAX_HALTING_VEHICLES for e in self.all_observed_edges
        ]
        
        MAX_PHASES_NUM = 10

        current_phases = [traci.trafficlight.getPhase(tls_id) / MAX_PHASES_NUM for tls_id in self.tls_ids]
        
        # Combine the observations into a single array
        #obs = np.array(halting_vehicles + current_phases, dtype=np.float32)
        obs = np.concatenate([halting_vehicles, current_phases]).astype(np.float32)        
        return obs


    # Old reward function, commented out for now, trying a simpler version.

    # def _get_reward(self):
    #     halted = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges)

    #     braking_penalty = 0
    #     slow_penalty = 0
    #     speed_sum = 0
    #     vehicle_count = 0

    #     for v in traci.vehicle.getIDList():
    #         accel = traci.vehicle.getAcceleration(v)
    #         speed = traci.vehicle.getSpeed(v)

    #         if accel < -3.0:
    #             braking_penalty += 1
    #         if speed < 2.0:
    #             slow_penalty += 1

    #         speed_sum += speed
    #         vehicle_count += 1

    #     avg_speed = speed_sum / vehicle_count if vehicle_count > 0 else 0
    #     avg_halted = halted / vehicle_count if vehicle_count > 0 else 0
    #     avg_slow = slow_penalty / vehicle_count if vehicle_count > 0 else 0

    #     arrived = traci.simulation.getArrivedNumber()
    #     teleports = traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber()
       
    #     reward = (
    #         + 0.5 * arrived
    #         + 0.5 * avg_speed
    #         - 2.0 * avg_halted
    #         - 1.5 * avg_slow
    #         - 2.0 * teleports
    #     )

    #    return reward

    def _get_reward(self):
        # Penalize the agent for the number of vehicles waiting at intersections
        total_halting_vehicles = sum(
            traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges
        )
        congestion_penalty = -total_halting_vehicles
            
            # --- 2. Reward for Throughput ---
        current_arrived_count = traci.simulation.getArrivedNumber()
        newly_arrived = current_arrived_count - self.last_arrived_count
        self.last_arrived_count = current_arrived_count
        throughput_reward = newly_arrived * 10.0 # Give a bonus for each car that arrives

        # --- 3. Penalty for Gridlock ---
        current_teleport_count = traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber()
        newly_teleported = current_teleport_count - self.last_teleport_count
        self.last_teleport_count = current_teleport_count
        gridlock_penalty = -newly_teleported * 20.0 # Give a large penalty for teleports

        # We combine the components and then scale the total down to prevent instability.
        combined_reward = congestion_penalty + throughput_reward + gridlock_penalty
        
        # Scale the final reward to a small range (e.g., around -1 to +1)
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

