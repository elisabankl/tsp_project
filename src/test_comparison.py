import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from customenv import CustomEnv
from sb3_contrib import MaskablePPO
from evaluate_greedy import greedy_action
from or_tools_google import main as or_tools_solver

# Function to evaluate the RL agent
def evaluate_rl_agent(env, model, num_instances=100):
    rl_rewards = []
    for _ in range(num_instances):
        obs, _ = env.reset(fixed_instance=False)
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            action_masks = env.action_masks()
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards
        rl_rewards.append(episode_rewards)
    return rl_rewards

# Function to evaluate the Greedy algorithm
def evaluate_greedy(env, num_instances=100):
    greedy_rewards = []
    for _ in range(num_instances):
        obs, _ = env.reset(fixed_instance=False)
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            action = greedy_action(env)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards
        greedy_rewards.append(episode_rewards)

        obs, _ = env.reset(fixed_instance=True)  # Reset the environment if done
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            action_masks = env.action_masks()
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards
        rl_rewards.append(episode_rewards)
        obs, _ = env.reset(fixed_instance=True)  # Reset the environment if done


    return greedy_rewards

# Function to evaluate the OR-Tools algorithm
def evaluate_or_tools(env, num_instances=100):
    or_tools_rewards = []
    for _ in range(num_instances):
        # Use OR-Tools to solve the instance
        or_tools_reward = or_tools_solver()
        or_tools_rewards.append(or_tools_reward)
    return or_tools_rewards

# Main script
if __name__ == "__main__":
    # Initialize the environment and load the RL model
    env = CustomEnv(25)
    model = MaskablePPO.load("masked_ppo_tsp_non_normalized_25", verbose=0)

    # Evaluate the RL agent
    print("Evaluating RL agent...")
    rl_rewards = evaluate_rl_agent(env, model)

    # Evaluate the Greedy algorithm
    print("Evaluating Greedy algorithm...")
    greedy_rewards = evaluate_greedy(env)

    # Evaluate the OR-Tools algorithm
    print("Evaluating OR-Tools algorithm...")
    or_tools_rewards = evaluate_or_tools(env)

    # Prepare data for plotting
    data = pd.DataFrame({
        "RL Agent": rl_rewards,
        "Greedy Algorithm": greedy_rewards,
        "OR-Tools Algorithm": or_tools_rewards
    })

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.boxplot([data["RL Agent"], data["Greedy Algorithm"], data["OR-Tools Algorithm"]],
                labels=["RL Agent", "Greedy Algorithm", "OR-Tools Algorithm"])
    plt.title("Performance Comparison Across 100 Instances")
    plt.ylabel("Total Reward")
    plt.xlabel("Algorithm")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print(data.describe())