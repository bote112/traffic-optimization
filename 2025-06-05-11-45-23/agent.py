import os
import time
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from opti_env import SumoEnv

# === Config ===
SUMO_CFG_PATH = "E:/licenta/oras-mediu/2025-06-05-11-45-23/osm.sumocfg"
LOG_DIR = "E:/licenta/oras-mediu/2025-06-05-11-45-23/ppo_logs"
MODEL_DIR = "E:/licenta/oras-mediu/2025-06-05-11-45-23/ppo_models"
TIMESTEPS = 2_000_000
SAVE_FREQ = 5000

# === Custom Callback for Autosaving ===
class SaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_file = os.path.join(self.save_path, f"ppo_sumo_{timestamp}")
            self.model.save(model_file)
            if self.verbose > 0:
                print(f"\n[AutoSave] Saved model at step {self.num_timesteps} to: {model_file}")
        return True

# # === Custom TensorBoard Logging Callback ===
# class TensorboardLoggingCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)

#     def _on_step(self) -> bool:
#         import traci

#         reward = self.locals.get("rewards", [None])[0]
#         if reward is not None:
#             self.logger.record("env/reward", reward)

#         vehicle_stats = {
#             "passenger": {"count": 0, "speed": 0.0, "wait": 0.0},
#             "pedestrian": {"count": 0, "speed": 0.0, "wait": 0.0},
#             "delivery": {"count": 0, "speed": 0.0, "wait": 0.0},
#             "emergency": {"count": 0, "speed": 0.0, "wait": 0.0},
#             "bicycle": {"count": 0, "speed": 0.0, "wait": 0.0},
#             "motorcycle": {"count": 0, "speed": 0.0, "wait": 0.0},
#             "truck": {"count": 0, "speed": 0.0, "wait": 0.0},
#         }

#         try:
#             for veh_id in traci.vehicle.getIDList():
#                 vtype = traci.vehicle.getTypeID(veh_id)
#                 speed = traci.vehicle.getSpeed(veh_id)
#                 wait = traci.vehicle.getWaitingTime(veh_id)

#                 if vtype in vehicle_stats:
#                     stats = vehicle_stats[vtype]
#                     stats["count"] += 1
#                     stats["speed"] += speed
#                     stats["wait"] += wait

#             for vtype, stats in vehicle_stats.items():
#                 count = stats["count"]
#                 if count > 0:
#                     avg_speed = stats["speed"] / count
#                     avg_wait = stats["wait"] / count
#                     self.logger.record(f"custom/avg_speed/{vtype}", avg_speed)
#                     self.logger.record(f"custom/avg_wait/{vtype}", avg_wait)

#             # Log teleports (sum of start and end teleports)
#             start_teleports = traci.simulation.getStartingTeleportNumber()
#             end_teleports = traci.simulation.getEndingTeleportNumber()
#             self.logger.record("custom/teleports", start_teleports + end_teleports)

#         except Exception as e:
#             if self.verbose:
#                 print("[LoggingCallback Error]", str(e))

#         return True

# === Create Environment ===
def make_env():
    env = SumoEnv(sumo_cfg=SUMO_CFG_PATH, gui=False)
    env = Monitor(env)  # Required for TensorBoard to track episode rewards
    return env

if __name__ == "__main__":

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    num_cpu = 6
    env = SubprocVecEnv([make_env for i in range(num_cpu)])

    policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]) # pi=policy network, vf=value network
)
    # === Create and Train PPO Model ===
    model = MultiOutputPPO(
        policy="MultiOutputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        # --- New Hyperparameters ---
        n_steps=4096,               # More data per update for stability
        learning_rate=1e-4,         # Smaller updates for more cautious learning
        policy_kwargs=policy_kwargs,  # A bigger brain for the agent
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,              
        clip_range=0.2,
        n_epochs=10
    )

    # Combine callbacks
    callback = CallbackList([
        SaveCallback(SAVE_FREQ, MODEL_DIR)
    ])

    print("Starting PPO training...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback,
    )
    print("Training complete.")

    # === Final Save ===
    final_timestamp = time.strftime("%Y%m%d-%H%M%S")
    final_model_path = os.path.join(MODEL_DIR, f"ppo_sumo_final_{final_timestamp}")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
