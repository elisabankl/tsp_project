import numpy as np
from customenv import CustomEnv
import matplotlib.pyplot as plt
import pandas as pd

def greedy_action(env):
    """Select the legal action with the highest immediate reward."""
    best_action = None
    best_reward = -np.inf
    for action in range(env.action_space.n):
        if env.allowed_actions[action]:
            if env.current_node is None:
                reward = -float(env.cost_matrix[action, action])  # Cost for selecting the first node
            else:
                reward = -float(env.distance_matrix[env.current_node, action])
            if reward > best_reward:
                best_reward = reward
                best_action = action
    return best_action

# Create the environment
env = CustomEnv(25,"Euclidic", normalize_rewards=True, p=0.1)

# Evaluate the greedy algorithm
obs, _ = env.reset()  # Extract the observation from the tuple
total_rewards = []
for episode in range(1000):
    dones = False
    truncated = False
    episode_rewards = 0
    while not (dones or truncated):
        action = greedy_action(env)
        obs, rewards, dones, truncated, info = env.step(action)
        episode_rewards += rewards
        if dones or truncated:
            total_rewards.append(episode_rewards)
            obs, _ = env.reset()  # Reset the environment if done

# Plot the results
print("Mean reward: ", np.mean(total_rewards))

plt.figure(figsize=(12, 6))

# Plot total rewards
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode (Greedy Algorithm)')

plt.tight_layout()
plt.show()