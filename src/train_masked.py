import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from matrices import generate_distance_matrix, generate_precedence_matrix, generate_cost_matrix
from customenv import CustomEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Function to apply action masks
def mask_fn(env):
    return env.env.action_masks()

# Create the training environment and wrap it with Monitor and ActionMasker
env = ActionMasker(Monitor(CustomEnv(25), filename="masked_ppo_25_non_normalized_monitor.csv", allow_early_resets=True,override_existing=False), mask_fn)
#env = DummyVecEnv([lambda: ActionMasker(Monitor(CustomEnv(25), filename="masked_ppo_25_non_normalized_monitor.csv", allow_early_resets=True), mask_fn)])
#env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create the evaluation environment and wrap it with Monitor and ActionMasker
eval_env = ActionMasker(Monitor(CustomEnv(25), filename="masked_ppo_eval_non_normalized_monitor.csv", allow_early_resets=True,override_existing=False), mask_fn)

#eval_env = DummyVecEnv([lambda: ActionMasker(Monitor(CustomEnv(25), filename="masked_ppo_eval_non_normalized_monitor.csv", allow_early_resets=True), mask_fn)])
#eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

# Set up the MaskableEvalCallback
eval_callback = MaskableEvalCallback(eval_env, best_model_save_path='./logs_non_normalized/',
                                     log_path='./logs_non_normalized/', eval_freq=50000,
                                     n_eval_episodes=200, deterministic=True, render=False)

# Load the existing model if it exists, otherwise create a new one
try:
    model = MaskablePPO.load("masked_ppo_tsp_non_normalized_25", env=env, verbose=0)
    print("Loaded existing model.")
except FileNotFoundError:
    model = MaskablePPO('MlpPolicy', env, verbose=0)
    print("Created new model.")

# Further train the model
model.learn(total_timesteps=2000000, callback=eval_callback)

# Save the model and normalization statistics
model.save("masked_ppo_tsp_non_normalized_25")
#env.save("vec_normalize.pkl")