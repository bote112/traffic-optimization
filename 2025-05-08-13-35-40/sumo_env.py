import gym
import numpy as np
import os
import subprocess
import traci
from gym import spaces

class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg, gui=False):
        super(SumoEnv, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.step_count = 0
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"


        # Start SUMO temporarily to fetch TLS info and all edges
        traci.start([self.sumo_binary, "-c", self.sumo_cfg])

        self.tls_ids = traci.trafficlight.getIDList()
        self.controlled_edges_map = {}
        self.phase_counts = []  # NEW: phases per TLS
        self.min_phase_duration = 100  # You can tune this value
        self.last_phase_change_step = {tls_id: -9999 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}
        
        self.decision_interval = 100  # Every 10 seconds in sim time (0.1s step)
        
        self.yellow_duration = 30  # 30 steps = 3 seconds if step length is 0.1
        self.in_yellow_phase = {tls_id: False for tls_id in self.tls_ids}
        self.yellow_remaining = {tls_id: 0 for tls_id in self.tls_ids}
        self.next_phase = {tls_id: 0 for tls_id in self.tls_ids}



        for tls_id in self.tls_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_edges = list(set([lane.split("_")[0] for lane in controlled_lanes]))
            self.controlled_edges_map[tls_id] = controlled_edges

            num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
            self.phase_counts.append(num_phases)

        all_edges = traci.edge.getIDList()
        self.all_observed_edges = [e for e in all_edges if not e.startswith(":")]

        traci.close()

        # ACTION SPACE: one discrete action per TLS
        self.action_space = spaces.MultiDiscrete(self.phase_counts)

        # OBSERVATION SPACE: vehicle counts per edge
        self.observation_space = spaces.Box(low=0, high=1000, shape=(len(self.all_observed_edges),), dtype=np.float32)

    def reset(self):
        if traci.isLoaded():
            traci.close()

        self.step_count = 0

        # Regenerate route file
        generator_path = os.path.join(os.path.dirname(__file__), "simulation_generator.py")
        route_file = os.path.join(os.path.dirname(__file__), "routes.rou.xml")
        subprocess.run(["python", generator_path, route_file], check=True)

        traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        return self._get_observation()

    def step(self, action):
        for i, tls_id in enumerate(self.tls_ids):
            num_phases = self.phase_counts[i]
            current_step = self.step_count

            if self.in_yellow_phase[tls_id]:
                # Countdown yellow
                self.yellow_remaining[tls_id] -= 1
                if self.yellow_remaining[tls_id] <= 0:
                    # Switch to the new green phase
                    traci.trafficlight.setPhase(tls_id, self.next_phase[tls_id])
                    self.last_actions[tls_id] = self.next_phase[tls_id]
                    self.last_phase_change_step[tls_id] = current_step
                    self.in_yellow_phase[tls_id] = False
            else:
                time_since_last = current_step - self.last_phase_change_step[tls_id]
                if time_since_last >= self.min_phase_duration:
                    requested_phase = int(action[i]) % num_phases
                    if requested_phase != self.last_actions[tls_id]:
                        # Start yellow transition
                        yellow_phase = self._find_yellow_phase(tls_id)
                        if yellow_phase is not None:
                            traci.trafficlight.setPhase(tls_id, yellow_phase)
                            self.in_yellow_phase[tls_id] = True
                            self.yellow_remaining[tls_id] = self.yellow_duration
                            self.next_phase[tls_id] = requested_phase
                        else:
                            # No yellow found — fallback to direct switch
                            traci.trafficlight.setPhase(tls_id, requested_phase)
                            self.last_actions[tls_id] = requested_phase
                            self.last_phase_change_step[tls_id] = current_step
                    else:
                        traci.trafficlight.setPhase(tls_id, requested_phase)
                else:
                    traci.trafficlight.setPhase(tls_id, self.last_actions[tls_id])

        traci.simulationStep()
        self.step_count += 1

        obs = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        return obs, reward, done, {}
    
    def _find_yellow_phase(self, tls_id):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        for idx, phase in enumerate(logic.phases):
            if "y" in phase.state:
                return idx
        return None  # If no yellow defined

    def _get_observation(self):
        return np.array([
            traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges
        ], dtype=np.float32)

    def _get_reward(self):
        # === Count halted vehicles ===
        halted = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.all_observed_edges)

        # === Init accumulators ===
        braking_penalty = 0
        slow_penalty = 0
        speed_sum = 0
        vehicle_count = 0

        # === Vehicle-level metrics ===
        for v in traci.vehicle.getIDList():
            accel = traci.vehicle.getAcceleration(v)
            speed = traci.vehicle.getSpeed(v)

            if accel < -3.0:
                braking_penalty += 1
            if speed < 0.5:
                slow_penalty += 1

            speed_sum += speed
            vehicle_count += 1

        avg_speed = (speed_sum / vehicle_count) if vehicle_count > 0 else 0

        # === Reward vehicles that completed trip ===
        arrived = traci.simulation.getArrivedNumber()

        # === Penalize teleports ===
        teleports = (
            traci.simulation.getStartingTeleportNumber() +
            traci.simulation.getEndingTeleportNumber()
        )

        # === Final Reward ===
        reward = (
            + 2.0 * arrived                 # Encourage trip completions
            + 0.7 * avg_speed               # Encourage traffic flow
            - 0.2 * halted                  # Discourage congestion
            - 0.3 * braking_penalty         # Discourage aggressive stops
            - 0.2 * slow_penalty            # Discourage crawling
            - 0.7 * teleports               # Heavily penalize gridlocks
        )

        return reward


    def _is_done(self):
        MAX_STEPS = 5000  
        return traci.simulation.getMinExpectedNumber() == 0 or self.step_count >= MAX_STEPS

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def close(self):
        if traci.isLoaded():
            traci.close()
