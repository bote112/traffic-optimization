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
        self.phase_counts = []
        self.min_phase_duration = 50
        self.last_phase_change_step = {tls_id: -9999 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}

        for tls_id in self.tls_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_edges = list(set([lane.split("_")[0] for lane in controlled_lanes]))
            self.controlled_edges_map[tls_id] = controlled_edges

            num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
            self.phase_counts.append(num_phases)

        all_edges = traci.edge.getIDList()
        self.all_observed_edges = [e for e in all_edges if not e.startswith(":")]

        traci.close()

        self.action_space = spaces.MultiDiscrete(self.phase_counts)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(len(self.all_observed_edges),), dtype=np.float32)

    def reset(self):
        if traci.isLoaded():
            traci.close()

        self.step_count = 0

        generator_path = os.path.join(os.path.dirname(__file__), "simulation_generator.py")
        route_file = os.path.join(os.path.dirname(__file__), "routes.rou.xml")
        subprocess.run(["python", generator_path, route_file], check=True)

        traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        return self._get_observation()

    def step(self, action):
        for i, tls_id in enumerate(self.tls_ids):
            num_phases = self.phase_counts[i]
            current_step = self.step_count
            time_since_last = current_step - self.last_phase_change_step[tls_id]

            if time_since_last >= self.min_phase_duration:
                requested_phase = int(action[i]) % num_phases
                if requested_phase != self.last_actions[tls_id]:
                    traci.trafficlight.setPhase(tls_id, requested_phase)
                    self.last_actions[tls_id] = requested_phase
                    self.last_phase_change_step[tls_id] = current_step

        traci.simulationStep()
        self.step_count += 1

        obs = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        return obs, reward, done, {}

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
            if speed < 0.5:
                slow_penalty += 1

            speed_sum += speed
            vehicle_count += 1

        avg_speed = (speed_sum / vehicle_count) if vehicle_count > 0 else 0
        arrived = traci.simulation.getArrivedNumber()
        teleports = traci.simulation.getStartingTeleportNumber() + traci.simulation.getEndingTeleportNumber()

        reward = (
            + 2.0 * arrived
            + 0.7 * avg_speed
            - 0.2 * halted
            - 0.3 * braking_penalty
            - 0.2 * slow_penalty
            - 0.7 * teleports
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
