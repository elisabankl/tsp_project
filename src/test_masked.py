import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from customenv import CustomEnv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

def generate_random_solution(env):
    obs, _ = env.reset(fixed_instance=True)
    done = False
    truncated = False
    episode_rewards = 0
    while not (done or truncated):
        # Get the action masks
        action_masks = env.action_masks()
        # Randomly select an allowed action
        allowed_actions = np.flatnonzero(action_masks)
        action = np.random.choice(allowed_actions)
        obs, rewards, done, truncated, info = env.step(action)
        episode_rewards += rewards
    return episode_rewards

# Load the trained Maskable PPO model
model = MaskablePPO.load("masked_ppo_tsp_reduced_state_rep_25", verbose=0)

# Initialize lists to store rewards for all instances
all_agent_rewards = []
all_random_rewards = []
all_deterministic_rewards = []
all_instance_labels = []

env = CustomEnv(25)


# Run the agent and random solutions for 10 different instances
for instance in range(10):
    obs, _ = env.reset(fixed_instance=False)  # Extract the observation from the tuple

    # Generate solutions using the trained agent (non-deterministic)
    agent_rewards = []
    for episode in range(100):
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            # Get the action masks
            action_masks = env.action_masks()
            # Predict the action using the action masks
            action, _states = model.predict(obs, deterministic=False, action_masks=action_masks)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards

            if done or truncated:
                agent_rewards.append(episode_rewards)
                obs, _ = env.reset(fixed_instance=True)  # Reset the environment if done

    # Generate solutions using the trained agent (deterministic)
    deterministic_rewards = []
    for episode in range(1):  # Only one deterministic solution per instance
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            # Get the action masks
            action_masks = env.action_masks()
            # Predict the action using the action masks
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards

            if done or truncated:
                deterministic_rewards.append(episode_rewards)
                obs, _ = env.reset(fixed_instance=True)  # Reset the environment if done

    # Generate random solutions
    random_rewards = []
    for _ in range(100):
        random_reward = generate_random_solution(env)
        random_rewards.append(random_reward)

    # Store rewards and instance labels
    all_agent_rewards.extend(agent_rewards)
    all_random_rewards.extend(random_rewards)
    all_deterministic_rewards.extend(deterministic_rewards)
    all_instance_labels.extend([f'Instance {instance + 1}'] * 100)

# Prepare data for plotting
data = pd.DataFrame({
    'Reward': all_agent_rewards + all_random_rewards,
    'Type': ['Agent'] * len(all_agent_rewards) + ['Random'] * len(all_random_rewards),
    'Instance': all_instance_labels * 2
})

# Box plot of rewards
plt.figure(figsize=(16, 8))
sns.boxplot(x='Instance', y='Reward', hue='Type', data=data)

# Add deterministic solution indicators
for i, reward in enumerate(all_deterministic_rewards):
    plt.scatter(i, reward, color='red', marker='D', s=100, zorder=10, label='Agent Deterministic' if i == 0 else "")

plt.xlabel('Instance')
plt.ylabel('Reward')
plt.title('Reward Comparison Across Multiple Instances')
plt.legend(title='Type')
plt.tight_layout()
plt.show()
