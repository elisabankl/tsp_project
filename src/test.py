import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from customenv import CustomEnv
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
model = PPO.load("ppo_custom_tsp", verbose=0)

# Create the environment and wrap it with Monitor
env = CustomEnv(4)
env = Monitor(env, filename="test_monitor.csv", allow_early_resets=False)

# Use the model for predictions
obs, _ = env.reset()  # Extract the observation from the tuple
for _ in range(1):
    dones = False
    truncated = False
    while not (dones or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
        if dones or truncated:
            env.render()
            obs, _ = env.reset()  # Reset the environment if done

# Load the monitor logs
monitor_data = pd.read_csv("test_monitor.csv", skiprows=1)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot average reward
plt.subplot(1, 2, 1)
plt.plot(monitor_data['r'])
plt.axhline(y=-22, color='r', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward')

# Plot episode length
plt.subplot(1, 2, 2)
plt.plot(monitor_data['l'])
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Episode Length')

plt.tight_layout()
plt.show()



