import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from customenv import CustomEnv
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from or_tools_google import solve_google_or_with_greedy_solution
from compare_permutate import solve_tsp_cheapest_insertion


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



def evaluate_agent(model_path,instance_type, graph_size, p_values = [0,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.125,0.15,0.2,0.3,2]):

# Load the trained Maskable PPO model
    model = MaskablePPO.load(model_path, verbose=0)


    # Initialize lists to store average rewards and confidence intervals for each method
    average_agent_rewards = []
    average_greedy_rewards = []
    average_random_rewards = []
    greedy_action_pct = []
    agent_cis = []
    greedy_cis = []
    random_cis = []
    agent_stds = []
    greedy_stds = []
    random_stds = []
    identical_solutions_pct = []
    better_solutions_pct = []
    constraint_counts = []
    worse_solutions_pct = []


    raw_data_per_p = {}  # To store raw data for each p value

    # Evaluate for each value of p
    for p in p_values:
        print(f"Evaluating for p = {p}...")
        env = CustomEnv(graph_size,instance_type,p=p)

        agent_rewards = []
        greedy_rewards = []
        random_rewards = []
        agent_greedy_actions = []
        greedy_agent_same_tour = []

        contraint_nrs = []
        # Generate and solve a fix number of instances
        for instance in range(100):
            obs, _ = env.reset(fixed_instance=False)
            contraint_nrs.append(np.sum(env.precedence_matrix))
 
        # Solve using the greedy algorithm
            done = False
            truncated = False
            episode_rewards = 0
            greedy_actions = []
            while not (done or truncated):
                action = greedy_action(env)
                obs, rewards, done, truncated, info = env.step(action)
                episode_rewards += rewards
                greedy_actions.append(action)
            greedy_rewards.append(episode_rewards)

        # Solve using the agent (deterministic)
            obs, _ = env.reset(fixed_instance=True)
            agent_actions = []

            done = False
            truncated = False
            episode_rewards = 0
            greedy_actions_count = 0
            while not (done or truncated):
                action_masks = env.action_masks()
                action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                if action == greedy_action(env):
                    greedy_actions_count += 1
                obs, rewards, done, truncated, info = env.step(action)
                episode_rewards += rewards
                agent_actions.append(action)
            agent_rewards.append(episode_rewards)
            agent_greedy_actions.append(greedy_actions_count)

            if greedy_actions == agent_actions:
                greedy_agent_same_tour.append(1)
            else:
                greedy_agent_same_tour.append(0)

        # Solve using the random algorithm
            random_reward = generate_random_solution(env)
            random_rewards.append(random_reward)

    # Calculate average rewards and confidence intervals for this value of p
        average_agent_rewards.append(np.mean(agent_rewards))
        average_greedy_rewards.append(np.mean(greedy_rewards))
        average_random_rewards.append(np.mean(random_rewards))
        greedy_action_pct.append(np.mean(agent_greedy_actions) / 25 * 100)


    #Calculate percentage of greedy tours:
        identical_solutions_pct.append(np.mean(greedy_agent_same_tour) * 100 ) # Percentage of tours that are identical to greedy

    #Calculate percentage of tours that are better than greedy
        better_solutions_pct.append(np.mean(np.array(agent_rewards) > np.array(greedy_rewards)) * 100 ) # Percentage of tours better than greedy

        worse_solutions_pct.append(np.mean(np.array(agent_rewards) < np.array(greedy_rewards)) * 100 ) # Percentage of tours worse than greedy

    # Calculate standard error of the mean (SEM) and confidence intervals
        agent_cis.append(1.96 * np.std(agent_rewards) / np.sqrt(len(agent_rewards)))
        greedy_cis.append(1.96 * np.std(greedy_rewards) / np.sqrt(len(greedy_rewards)))
        random_cis.append(1.96 * np.std(random_rewards) / np.sqrt(len(random_rewards)))


    # Calculate standard deviations
        agent_stds.append(np.std(agent_rewards))
        greedy_stds.append(np.std(greedy_rewards))
        random_stds.append(np.std(random_rewards))

        constraint_counts.append(np.mean(contraint_nrs))  # Average number of precedence constraints


        raw_data_per_p[p]  ={
        'agent_rewards': agent_rewards,
        'greedy_rewards': greedy_rewards,
        'random_rewards': random_rewards,
        }

    # Create a new data structure to store rewards and statistics
    p_results = {}
    for p_idx, p in enumerate(p_values):
        # Create a dictionary for this p value
        p_results[p] = {
            'agent_rewards': average_agent_rewards[p_idx],
            'greedy_rewards': average_greedy_rewards[p_idx],
            'random_rewards': average_random_rewards[p_idx],
            'agent_std': agent_stds[p_idx],
            'greedy_std': greedy_stds[p_idx],
            'random_std': random_stds[p_idx],
            'greedy_actions_pct': greedy_action_pct[p_idx],
            'agent_vs_greedy_diff': average_agent_rewards[p_idx] - average_greedy_rewards[p_idx],
            "identical_solutions_pct": identical_solutions_pct[p_idx],
            "better_solutions_pct": better_solutions_pct[p_idx],
            "worse_solutions_pct": worse_solutions_pct[p_idx],
            "constraint_count": constraint_counts[p_idx]
        }


    agent_rewards_list = [p_results[p]['agent_rewards'] for p in p_values]
    greedy_rewards_list = [p_results[p]['greedy_rewards'] for p in p_values]
    random_rewards_list = [p_results[p]['random_rewards'] for p in p_values]


    agent_stds_list = [p_results[p]['agent_std'] for p in p_values]
    greedy_stds_list = [p_results[p]['greedy_std'] for p in p_values]
    random_stds_list = [p_results[p]['random_std'] for p in p_values]

# Make sure correlations_per_p is defined before using it
    correlations_per_p = {}

    # Calculate correlations for each p value
    for p in p_values:
        if p in raw_data_per_p:  # Make sure we have data for this p value
            data = raw_data_per_p[p]
        
        # Calculate Pearson correlations
            try:
                agent_greedy_corr, agent_greedy_p = stats.pearsonr(data['agent_rewards'], data['greedy_rewards'])
                agent_random_corr, agent_random_p = stats.pearsonr(data['agent_rewards'], data['random_rewards'])
                greedy_random_corr, greedy_random_p = stats.pearsonr(data['greedy_rewards'], data['random_rewards'])
            
                correlations_per_p[p] = {
                'agent_greedy': agent_greedy_corr,
                'agent_random': agent_random_corr,
                'greedy_random': greedy_random_corr,
                'agent_greedy_p': agent_greedy_p,
                'agent_random_p': agent_random_p,
                'greedy_random_p': greedy_random_p
            }
            except:
            # If correlation calculation fails, use default values
                correlations_per_p[p] = {
                'agent_greedy': 0.0,
                'agent_random': 0.0,
                'greedy_random': 0.0,
                'agent_greedy_p': 1.0,
                'agent_random_p': 1.0,
                'greedy_random_p': 1.0
            }
        else:
            # If no data available, use default values
            correlations_per_p[p] = {
            'agent_greedy': 0.0,
            'agent_random': 0.0,
            'greedy_random': 0.0,
            'agent_greedy_p': 1.0,
            'agent_random_p': 1.0,
            'greedy_random_p': 1.0
        }
#create summary table with correlations and other metrics
    summary_data = {
    "p_values": p_values,
    "agent_rewards": [p_results[p]['agent_rewards'] for p in p_values],
    "greedy_rewards": [p_results[p]['greedy_rewards'] for p in p_values],
    "random_rewards": [p_results[p]['random_rewards'] for p in p_values],
    "agent_std": [p_results[p]['agent_std'] for p in p_values],
    "greedy_std": [p_results[p]['greedy_std'] for p in p_values],
    "random_std": [p_results[p]['random_std'] for p in p_values],
    "greedy_actions_pct": [p_results[p]['greedy_actions_pct'] for p in p_values],
    "agent_vs_greedy_diff": [p_results[p]['agent_vs_greedy_diff'] for p in p_values],
    "identical_solutions_pct": identical_solutions_pct,
    "better_solutions_pct": better_solutions_pct,
    "worse_solutions_pct": worse_solutions_pct,
    "constraint_count": constraint_counts,
    "agent_greedy_corr": [correlations_per_p[p]['agent_greedy'] for p in p_values],
    "agent_random_corr": [correlations_per_p[p]['agent_random'] for p in p_values],
    "greedy_random_corr": [correlations_per_p[p]['greedy_random'] for p in p_values]
}

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(3))
    summary_df.to_csv("evaluation_summary_with_correlations.csv", index=False)


# Create a comprehensive figure with 8 subplots
    plt.figure(figsize=(22, 30))

# 1. Plot comparing average rewards with confidence intervals
    plt.subplot(4, 2, 1)
    plt.title("Algorithm Performance with 95% Confidence Intervals", pad=15)


    plt.plot(p_values, agent_rewards_list, label="Agent", marker="o", color="blue")
    plt.plot(p_values, greedy_rewards_list, label="Nearest Neighbor", marker="s", color="green")
    plt.plot(p_values, random_rewards_list, label="Random", marker="^", color="orange")

    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)

    plt.fill_between(p_values,
                np.array(agent_rewards_list) - np.array(agent_cis),
                np.array(agent_rewards_list) + np.array(agent_cis),
                color="blue", alpha=0.3)

    plt.fill_between(p_values,
                np.array(greedy_rewards_list) - np.array(greedy_cis),
                np.array(greedy_rewards_list) + np.array(greedy_cis),
                color="green", alpha=0.3)

    plt.fill_between(p_values,
                np.array(random_rewards_list) - np.array(random_cis),
                np.array(random_rewards_list) + np.array(random_cis),
                color="orange", alpha=0.3)

# 2. Plot comparing rewards with standard deviations
    plt.subplot(4, 2, 2)
    plt.title("Algorithm Performance with Standard Deviations", pad=15)


    plt.plot(p_values, agent_rewards_list, label="Agent", marker="o", color="blue")
    plt.fill_between(p_values,
                np.array(agent_rewards_list) - np.array(agent_stds_list),
                np.array(agent_rewards_list) + np.array(agent_stds_list),
                color="blue", alpha=0.2)

    plt.plot(p_values, greedy_rewards_list, label="Greedy", marker="s", color="green")
    plt.fill_between(p_values,
                np.array(greedy_rewards_list) - np.array(greedy_stds_list),
                np.array(greedy_rewards_list) + np.array(greedy_stds_list),
                color="green", alpha=0.2)

    plt.plot(p_values, random_rewards_list, label="Random", marker="^", color="orange")
    plt.fill_between(p_values,
                np.array(random_rewards_list) - np.array(random_stds_list),
                np.array(random_rewards_list) + np.array(random_stds_list),
                color="orange", alpha=0.2)

    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()    
    plt.grid(True)

# 3. Plot showing agent vs greedy difference
    plt.subplot(4, 2, 3)
    plt.title("Agent vs Greedy Average Reward Difference")
    agent_vs_greedy = [p_results[p]['agent_vs_greedy_diff'] for p in p_values]
    plt.plot(p_values, agent_vs_greedy, color="purple", alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Agent Reward - Greedy Reward")
    plt.grid(True, axis='y')

# 4. Plot showing percentage of agent solutions identical to and better than greedy
    plt.subplot(4, 2, 4)
    plt.title("Agent vs Greedy Solution Comparison")
    plt.plot(p_values, [p_results[p]['greedy_actions_pct'] for p in p_values], label="% Greedy Actions", marker="*", color="red")
    plt.plot(p_values, [p_results[p]['identical_solutions_pct'] for p in p_values], label="% Identical Solutions", marker="o", color="purple")
    plt.plot(p_values, [p_results[p]['better_solutions_pct'] for p in p_values], label="% Better Than Greedy", marker="^", color="green")
    plt.plot(p_values, [p_results[p]['worse_solutions_pct'] for p in p_values], label="% Worse Than Greedy", marker="s", color="orange")
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

# 5. Fixed plot showing average number of precedence constraints
    plt.subplot(4, 2, 5)
    plt.title("Average Number of Precedence Constraints")
    plt.plot(p_values, constraint_counts, marker="o", color="brown")
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Constraint Count")
    plt.grid(True)

# 6. Plot showing ratio of agent reward to greedy reward
    plt.subplot(4, 2, 6)
    plt.title("Agent/Greedy Reward Ratio")
    reward_ratios = []  
    for i in range(len(p_values)):
        # Avoid division by zero or very small numbers
        if greedy_rewards_list[i] != 0 and abs(greedy_rewards_list[i]) > 1e-6:
            ratio = agent_rewards_list[i] / greedy_rewards_list[i]
            reward_ratios.append(ratio)
        else:
            reward_ratios.append(1.0)  # Default to 1.0 if greedy reward is zero

    plt.plot(p_values, reward_ratios, marker="o", color="darkblue")
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Agent Reward / Greedy Reward")
    plt.grid(True)

# 7. Add correlation plot
    plt.subplot(4, 2, 7)
    plt.title("Correlation Between Solution Methods")

# Extract correlation data
    agent_greedy_corrs = [correlations_per_p[p]['agent_greedy'] for p in p_values]
    agent_random_corrs = [correlations_per_p[p]['agent_random'] for p in p_values]
    greedy_random_corrs = [correlations_per_p[p]['greedy_random'] for p in p_values]

    plt.plot(p_values, agent_greedy_corrs, label="Agent vs Greedy", marker="o", color="blue")
    plt.plot(p_values, agent_random_corrs, label="Agent vs Random", marker="s", color="green")
    plt.plot(p_values, greedy_random_corrs, label="Greedy vs Random", marker="^", color="red")

    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.legend()    
    plt.grid(True)

# 8. Improve the box plot visualization 
    plt.subplot(4, 2, 8)
    plt.title("Agent vs Greedy Reward Difference", pad=15)  # Better title with padding


# Prepare data for box plots
    box_data = []
    box_labels = []
    box_positions = []
    box_colors = []

    pos = 0

    for p_idx, p in enumerate(p_values):
        data = raw_data_per_p[p]
    
    # Add greedy data
        box_data.append(np.array(data["agent_rewards"])-np.array(data['greedy_rewards']))
        box_labels.append(f'{p}')
        box_positions.append(pos)
        box_colors.append('lightgreen')
        pos += 1
    
    # Add some spacing between p-values
        pos += 0.5

# Create box plots
    bp = plt.boxplot(box_data, positions=box_positions, patch_artist=True, widths=0.6)

# Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    plt.xticks(box_positions, box_labels, ha='right')  # Rotate labels
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Agent Reward - Greedy Reward")

# Add a horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    model_name = model_path.split(".")[0]  # Remove the .zip extension

    # Create a more informative filename
    output_filename = f"comprehensive_evaluation_{model_name}_{instance_type}.png"

    # Save with the new filename
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Also update the CSV filename
    csv_filename = f"evaluation_summary_{model_name}_{instance_type}.csv"
    summary_df.to_csv(csv_filename, index=False)