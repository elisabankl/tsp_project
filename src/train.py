import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from matrices import generate_distance_matrix, generate_precedence_matrix, generate_cost_matrix
from customenv import CustomEnv


# Create the environment and wrap it with Monitor
env = CustomEnv(4)
env = Monitor(env, filename="monitor.csv", allow_early_resets=True)
check_env(env)

# Load the existing model if it exists, otherwise create a new one
try:
    model = PPO.load("ppo_custom_tsp", env=env)
    print("Loaded existing model.")
except FileNotFoundError:
    model = PPO('MlpPolicy', env, verbose=1)
    print("Created new model.")

# Further train the model
model.learn(total_timesteps=300000)

# Save the model
model.save("ppo_custom_tsp")