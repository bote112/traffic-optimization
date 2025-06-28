import os
import time
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from opti_env2 import SumoEnv 
from stable_baselines3.common.logger import configure


# === CONFIGURATION ---

# --- MUST-CHANGE: Set the exact path to the .zip file of your best "safe" model ---
PHASE_1_MODEL_PATH = "E:/licenta/oras-mediu/2025-06-05-11-45-23/ppo_models/ppo_sumo_20250614-085909.zip"

# --- Other paths and settings ---
BASE_PROJECT_DIR = "E:/licenta/oras-mediu/2025-06-05-11-45-23"
SUMO_CFG_PATH = os.path.join(BASE_PROJECT_DIR, "osm.sumocfg")

# New folders for this speed-training run
PHASE_2_LOG_DIR = os.path.join(BASE_PROJECT_DIR, "ppo_logs_phase2", "Run_Speed_01")
PHASE_2_MODEL_DIR = os.path.join(BASE_PROJECT_DIR, "ppo_models_phase2")

TIMESTEPS_TO_TRAIN = 1_000_000 # How many additional steps to train for speed
SAVE_FREQ = 5000

# === Custom Callback for Autosaving (Unchanged) ===
class SaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_file = os.path.join(self.save_path, f"ppo_sumo_phase2_{timestamp}")
            self.model.save(model_file)
            if self.verbose > 0:
                print(f"\n[AutoSave] Saved Phase 2 model at step {self.num_timesteps} to: {model_file}")
        return True

# === Create Environment Function (Unchanged) ===
def make_env():
    # This MUST create an env with your new Phase 2 reward function
    env = SumoEnv(sumo_cfg=SUMO_CFG_PATH, gui=False)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # --- SETUP ---
    os.makedirs(PHASE_2_LOG_DIR, exist_ok=True)
    os.makedirs(PHASE_2_MODEL_DIR, exist_ok=True)

    if not os.path.exists(PHASE_1_MODEL_PATH):
        print(f"\n[FATAL ERROR] Phase 1 model not found at: {PHASE_1_MODEL_PATH}")
        print("Please update the 'PHASE_1_MODEL_PATH' variable in this script.")
        exit()

    # --- ENVIRONMENT AND MODEL LOADING ---
    env = make_env()

    print(f"\n--- Loading pre-trained 'safe' model from: {os.path.basename(PHASE_1_MODEL_PATH)} ---")
    
    # Load the specified model. 
    model = MultiOutputPPO.load(PHASE_1_MODEL_PATH, env=env)
    
    new_logger = configure(PHASE_2_LOG_DIR, ["tensorboard"])
    model.set_logger(new_logger)


    # --- PHASE 2 TRAINING ---
    callback = CallbackList([SaveCallback(SAVE_FREQ, PHASE_2_MODEL_DIR)])

    print("\n--- Starting Phase 2 Training: Optimizing for Speed ---")
    model.learn(
        total_timesteps=TIMESTEPS_TO_TRAIN,
        callback=callback,
        # This ensures the step counter in the logs is continuous
        reset_num_timesteps=False 
    )
    print("\n--- Phase 2 Training complete. ---")

    # --- FINAL SAVE ---
    final_timestamp = time.strftime("%Y%m%d-%H%M%S")
    final_model_path = os.path.join(PHASE_2_MODEL_DIR, f"ppo_sumo_phase2_final_{final_timestamp}")
    model.save(final_model_path)
    print(f"Final Phase 2 model saved to: {final_model_path}")