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




    def create_comparison_plots(instance_type, p_values):
    """Create comprehensive comparison plots for all solvers."""
    # Add OR-Tools and AntColony to the list of solvers to plot
    results_root = os.path.join("results", instance_type)

    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'p', 'h', 'd']
    
    # First create the main comparison plots across all algorithms
    plt.figure(figsize=(24, 20))
    
    # 1. Plot comparing average rewards with confidence intervals
    plt.subplot(4, 1, 1)
    plt.title(f"Algorithm Performance Comparison ({instance_type}, {len(p_values)} p-values, {n_instances} runs)", pad=15)
    
    # Plot each solver
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['avg_rewards'],
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
        plt.fill_between(
            p_values,
            np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['confidence_intervals']),
            np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['confidence_intervals']),
            color=colors[color_idx], alpha=0.2
        )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)
    
    # 2. Plot comparing rewards with standard deviations
    plt.subplot(4, 1, 2)
    plt.title("Algorithm Performance with Standard Deviations", pad=15)
    
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['avg_rewards'], 
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
        plt.fill_between(
            p_values,
            np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['std_rewards']),
            np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['std_rewards']),
            color=colors[color_idx], alpha=0.2
        )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)
    
    # 3. Plot showing percentage of best solutions
    plt.subplot(4, 1, 3)
    plt.title("Percentage of Best Solutions by Algorithm", pad=15)
    
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['best_solution_pct'], 
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Best Solution Percentage")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    # 4. Heatmap of pairwise correlations at different p values
    plt.subplot(4, 1, 4)
    plt.title("Algorithm Correlation Heatmap", pad=15)
    
    # Create a subplot for each p value
    n_p_values = len(p_values)
    max_cols = min(4, n_p_values)  # Maximum 4 columns
    n_rows = (n_p_values + max_cols - 1) // max_cols
    
    # Calculate the best placement for the heatmap grid within the subplot
    grid_height = 0.7  # Height of the grid within the subplot
    grid_width = 0.9   # Width of the grid
    
    # For each p-value, create a correlation heatmap
    for p_idx, p in enumerate(p_values):
        ax = plt.gcf().add_subplot(n_rows, max_cols, p_idx+1)
        
        # Create the correlation matrix for this p value
        corr_matrix = np.zeros((len(all_solvers), len(all_solvers)))
        for i, name1 in enumerate(all_solvers):
            for j, name2 in enumerate(all_solvers):
                if i == j:
                    corr_matrix[i, j] = 1.0  # Correlation with self is 1
                elif p in correlations and name1 in correlations[p] and name2 in correlations[p][name1]:
                    corr_matrix[i, j] = correlations[p][name1][name2]
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add title
        ax.set_title(f"p = {p}")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_solvers)))
        ax.set_yticks(np.arange(len(all_solvers)))
        
        # Make labels more readable
        short_names = [name[:5] + '..' if len(name) > 7 else name for name in all_solvers]
        ax.set_xticklabels(short_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(short_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add correlation values as text
        for i in range(len(all_solvers)):
            for j in range(len(all_solvers)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", 
                              color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate title
    
    # 5. Plot showing difference between agent rewards and nearest neighbor rewards
    plt.figure(figsize=(24, 10))
    plt.title("Algorithm vs Nearest Neighbor Reward Difference", pad=15)
    
    # Plot each agent minus nearest neighbor
    for idx, name in enumerate(all_solvers):
        if name != 'NearestNeighbor':
            color_idx = idx % len(colors)
            marker_idx = idx % len(markers)
            
            # Calculate differences
            differences = []
            confidence_intervals = []
            
            for p_idx, p in enumerate(p_values):
                agent_reward = all_agents_data[name]['avg_rewards'][p_idx]
                nn_reward = all_agents_data['NearestNeighbor']['avg_rewards'][p_idx]
                diff = agent_reward - nn_reward
                
                # Calculate confidence interval for the difference
                agent_ci = all_agents_data[name]['confidence_intervals'][p_idx]
                nn_ci = all_agents_data['NearestNeighbor']['confidence_intervals'][p_idx]
                diff_ci = np.sqrt(agent_ci**2 + nn_ci**2)
                
                differences.append(diff)
                confidence_intervals.append(diff_ci)
            
            plt.plot(p_values, differences, 
                     label=f"{name} - NN", marker=markers[marker_idx], color=colors[color_idx])
            plt.fill_between(
                p_values,
                np.array(differences) - np.array(confidence_intervals),
                np.array(differences) + np.array(confidence_intervals),
                color=colors[color_idx], alpha=0.2
            )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Reward Difference")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Now create detailed plots for each agent (like in evaluate_agent)
    for agent_idx, agent_name in enumerate(agent_names):
        plt.figure(figsize=(22, 30))
        
        # Extract data for this agent
        agent_rewards = all_agents_data[agent_name]['avg_rewards']
        agent_cis = all_agents_data[agent_name]['confidence_intervals']
        agent_stds = all_agents_data[agent_name]['std_rewards']
        
        # Extract data for comparison methods
        greedy_rewards = all_agents_data['NearestNeighbor']['avg_rewards']
        greedy_cis = all_agents_data['NearestNeighbor']['confidence_intervals']
        greedy_stds = all_agents_data['NearestNeighbor']['std_rewards']

        ci_rewards = all_agents_data['Cheapest Insertion']['avg_rewards']
        ci_cis = all_agents_data['Cheapest Insertion']['confidence_intervals']
        ci_stds = all_agents_data['Cheapest Insertion']['std_rewards']
        
        # Extract OR-Tools data
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            ortools_rewards = all_agents_data['OR-Tools']['avg_rewards']
            ortools_cis = all_agents_data['OR-Tools']['confidence_intervals']
            ortools_stds = all_agents_data['OR-Tools']['std_rewards']
        
        # Extract Ant Colony data if available
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            antcolony_rewards = all_agents_data['AntColony']['avg_rewards']
            antcolony_cis = all_agents_data['AntColony']['confidence_intervals']
            antcolony_stds = all_agents_data['AntColony']['std_rewards']
        
        # 1. Plot comparing average rewards with confidence intervals
        plt.subplot(4, 2, 1)
        plt.title(f"{agent_name} Performance with 95% Confidence Intervals", pad=15)
        
        plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
        plt.plot(p_values, greedy_rewards, label="Nearest Neighbor", marker="s", color="green")

        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.plot(p_values, antcolony_rewards, label="Ant Colony", marker="^", color="purple")
        
        plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_cis),
                    np.array(agent_rewards) + np.array(agent_cis),
                    color="blue", alpha=0.3)
        
        plt.fill_between(p_values,
                    np.array(greedy_rewards) - np.array(greedy_cis),
                    np.array(greedy_rewards) + np.array(greedy_cis),
                    color="green", alpha=0.3)
        
        plt.plot(p_values, ci_rewards, label="Cheapest Insertion", marker="x", color="cyan")
        plt.fill_between(p_values,
            np.array(ci_rewards) - np.array(ci_cis),
            np.array(ci_rewards) + np.array(ci_cis),
            color="cyan", alpha=0.3)

        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_cis),
                    np.array(ortools_rewards) + np.array(ortools_cis),
                    color="red", alpha=0.3)
                    
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.fill_between(p_values,
                        np.array(antcolony_rewards) - np.array(antcolony_cis),
                        np.array(antcolony_rewards) + np.array(antcolony_cis),
                        color="purple", alpha=0.3)
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Reward")
        plt.ylim(None, 0)  # Negative rewards with 0 at top
        plt.legend()
        plt.grid(True)
        
        # 2. Plot comparing rewards with standard deviations
        plt.subplot(4, 2, 2)
        plt.title(f"{agent_name} Performance with Standard Deviations", pad=15)
        
        plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
        plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_stds),
                    np.array(agent_rewards) + np.array(agent_stds),
                    color="blue", alpha=0.2)
        
        plt.plot(p_values, greedy_rewards, label="Nearest Neighbor", marker="s", color="green")
        plt.fill_between(p_values,
                    np.array(greedy_rewards) - np.array(greedy_stds),
                    np.array(greedy_rewards) + np.array(greedy_stds),
                    color="green", alpha=0.2)
        
        plt.plot(p_values, ci_rewards, label="Cheapest Insertion", marker="x", color="cyan")
        plt.fill_between(p_values,
            np.array(ci_rewards) - np.array(ci_stds),
            np.array(ci_rewards) + np.array(ci_stds),
            color="cyan", alpha=0.2)
        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_stds),
                    np.array(ortools_rewards) + np.array(ortools_stds),
                    color="red", alpha=0.2)
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.plot(p_values, antcolony_rewards, label="Ant Colony", marker="^", color="purple")
            plt.fill_between(p_values,
                        np.array(antcolony_rewards) - np.array(antcolony_stds),
                        np.array(antcolony_rewards) + np.array(antcolony_stds),
                        color="purple", alpha=0.2)
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Reward")
        plt.ylim(None, 0)  # Negative rewards with 0 at top
        plt.legend()    
        plt.grid(True)

        # 3. Plot showing agent vs greedy difference
        plt.subplot(4, 2, 3)
        plt.title(f"{agent_name} vs Nearest Neighbor Average Reward Difference")
        agent_vs_greedy = np.array(agent_rewards) - np.array(greedy_rewards)
        plt.plot(p_values, agent_vs_greedy, color="purple", alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward - NN Reward")
        plt.grid(True, axis='y')
        
        # 4. Plot showing percentage of agent solutions better than greedy
        plt.subplot(4, 2, 4)
        plt.title(f"{agent_name} vs Nearest Neighbor Solution Comparison")
        
        # Calculate percentage of instances where agent is better
        better_than_nn = []
        identical_to_nn = []
        worse_than_nn = []
        greedy_action_pct = []
        
        for p_idx, p in enumerate(p_values):
            agent_raw_rewards = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw_rewards = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            
            better = np.sum(np.array(agent_raw_rewards) > np.array(nn_raw_rewards))
            identical = np.sum(np.array(agent_raw_rewards) == np.array(nn_raw_rewards))
            worse = np.sum(np.array(agent_raw_rewards) < np.array(nn_raw_rewards))
            
            total = len(agent_raw_rewards)
            better_than_nn.append((better / total) * 100 if total > 0 else 0)
            identical_to_nn.append((identical / total) * 100 if total > 0 else 0)
            worse_than_nn.append((worse / total) * 100 if total > 0 else 0)
            
            # Calculate average percentage of greedy choices if available
            if agent_name in all_agents_data and 'greedy_choice_pct' in all_agents_data[agent_name]:
                greedy_action_pct.append(all_agents_data[agent_name]['greedy_choice_pct'][p_idx])
            else:
                greedy_action_pct.append(0)  # Default if not available
        
        plt.plot(p_values, better_than_nn, label="% Better Than NN", marker="^", color="green")
        plt.plot(p_values, identical_to_nn, label="% Identical Solutions", marker="o", color="purple")
        plt.plot(p_values, worse_than_nn, label="% Worse Than NN", marker="s", color="orange")
        
        # Plot percentage of greedy actions if available
        if any(greedy_action_pct):
            plt.plot(p_values, greedy_action_pct, label="% Greedy Actions", marker="*", color="red")
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        # 5. Plot showing average number of precedence constraints
        plt.subplot(4, 2, 5)
        plt.title("Average Number of Precedence Constraints")
        
        # Calculate or extract constraint counts
        if 'constraint_counts' in all_agents_data:
            constraint_counts = all_agents_data['constraint_counts']
        else:
            constraint_counts = []
            for p_idx, p in enumerate(p_values):
                # Calculate the theoretical number of constraints based on p and graph size
                graph_size = all_agents_data[agent_name]['avg_rewards'][p_idx]
                theoretical_max = graph_size * (graph_size - 1) / 2  # Maximum possible constraints
                constraint_counts.append(p * theoretical_max)
                
        plt.plot(p_values, constraint_counts, marker="o", color="brown")
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Constraint Count")
        plt.grid(True)
        
        # 6. Plot showing ratio of agent reward to greedy reward
        plt.subplot(4, 2, 6)
        plt.title(f"{agent_name}/NN Reward Ratio")
        reward_ratios = []
        
        for i in range(len(p_values)):
            # Avoid division by zero or very small numbers
            if greedy_rewards[i] != 0 and abs(greedy_rewards[i]) > 1e-6:
                ratio = agent_rewards[i] / greedy_rewards[i]
                reward_ratios.append(ratio)
            else:
                reward_ratios.append(1.0)  # Default to 1.0 if greedy reward is zero
        
        plt.plot(p_values, reward_ratios, marker="o", color="darkblue")
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward / NN Reward")
        plt.grid(True)
        
        # 7. Add correlation plot
        plt.subplot(4, 2, 7)
        plt.title(f"Correlation of {agent_name} With Other Methods")
        
        # Extract correlation data for this agent with other methods
        corr_data = {}
        for other in all_solvers:
            if other != agent_name:
                corr_data[other] = []
                for p in p_values:
                    if p in correlations and agent_name in correlations[p] and other in correlations[p][agent_name]:
                        corr_data[other].append(correlations[p][agent_name][other])
                    else:
                        corr_data[other].append(0)
        
        # Plot correlation with each other method
        for idx, (other, corrs) in enumerate(corr_data.items()):
            color_idx = idx % len(colors)
            plt.plot(p_values, corrs, label=f"{agent_name} vs {other}", marker="o", color=colors[color_idx])
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.legend()
        plt.grid(True)
        
        # 8. Box plot of agent vs greedy reward difference
        plt.subplot(4, 2, 8)
        plt.title(f"{agent_name} vs NN Reward Difference Distribution", pad=15)
        
        # Prepare data for box plots
        box_data = []
        box_labels = []
        box_positions = []
        box_colors = []
        
        pos = 0
        for p_idx, p in enumerate(p_values):
            if p not in all_agents_data[agent_name]['raw_data_per_p']:
                continue
                
            agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            if or_tools_enabled:
                ort_raw = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
            # NN differences
            nn_diffs = np.array(agent_raw) - np.array(nn_raw)
            box_data[f'p{p}_vs_nn'] = nn_diffs.tolist()
            
            # OR-Tools differences
            if or_tools_enabled:
                min_len = min(len(agent_raw), len(ort_raw))
                if min_len > 0:
                    ort_diffs = np.array(agent_raw[:min_len]) - np.array(ort_raw[:min_len])
                    box_data[f'p{p}_vs_ort'] = ort_diffs.tolist()
        
        # Export as long format for easier analysis
        boxplot_rows = []
        for p in p_values:
            if f'p{p}_vs_nn' in box_data:
                for diff in box_data[f'p{p}_vs_nn']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_NN',
                        'reward_difference': diff
                    })
            
            if f'p{p}_vs_ort' in box_data:
                for diff in box_data[f'p{p}_vs_ort']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_OR_Tools',
                        'reward_difference': diff
                    })
        
        boxplot_df = pd.DataFrame(boxplot_rows)
        boxplot_df.to_csv(f"{agent_name}_boxplot_data_{instance_type}_{timestamp}.csv", index=False)
        print(f"{agent_name} boxplot data exported to {agent_name}_boxplot_data_{instance_type}_{timestamp}.csv")
    
    # Now call the plotting function as before
    create_comparison_plots(
        all_agents_data=all_agents_data,
        p_values=p_values,
        agent_names=agent_names,
        ant_solver_enabled=ant_solver_enabled,
        or_solver_enabled=or_tools_enabled,
        instance_type=instance_type,
        n_instances=n_instances,
        correlations=correlations
    )
    
    return all_agents_data, correlations, agent_names

    
# Add this function to create the plots at the end of evaluate_multiple_agents
def create_comparison_plots(all_agents_data, p_values, agent_names, ant_solver_enabled,or_solver_enabled, instance_type, n_instances, correlations):
    """Create comprehensive comparison plots for all solvers."""
    # Add OR-Tools and AntColony to the list of solvers to plot
    all_solvers = agent_names + ['NearestNeighbor', 'Cheapest Insertion']
    if or_solver_enabled:
        all_solvers.append('OR-Tools')
    if ant_solver_enabled:
        all_solvers.append('AntColony')
        
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'p', 'h', 'd']
    
    # First create the main comparison plots across all algorithms
    plt.figure(figsize=(24, 20))
    
    # 1. Plot comparing average rewards with confidence intervals
    plt.subplot(4, 1, 1)
    plt.title(f"Algorithm Performance Comparison ({instance_type}, {len(p_values)} p-values, {n_instances} runs)", pad=15)
    
    # Plot each solver
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['avg_rewards'],
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
        plt.fill_between(
            p_values,
            np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['confidence_intervals']),
            np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['confidence_intervals']),
            color=colors[color_idx], alpha=0.2
        )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)
    
    # 2. Plot comparing rewards with standard deviations
    plt.subplot(4, 1, 2)
    plt.title("Algorithm Performance with Standard Deviations", pad=15)
    
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['avg_rewards'], 
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
        plt.fill_between(
            p_values,
            np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['std_rewards']),
            np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['std_rewards']),
            color=colors[color_idx], alpha=0.2
        )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)
    
    # 3. Plot showing percentage of best solutions
    plt.subplot(4, 1, 3)
    plt.title("Percentage of Best Solutions by Algorithm", pad=15)
    
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['best_solution_pct'], 
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Best Solution Percentage")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    # 4. Heatmap of pairwise correlations at different p values
    plt.subplot(4, 1, 4)
    plt.title("Algorithm Correlation Heatmap", pad=15)
    
    # Create a subplot for each p value
    n_p_values = len(p_values)
    max_cols = min(4, n_p_values)  # Maximum 4 columns
    n_rows = (n_p_values + max_cols - 1) // max_cols
    
    # Calculate the best placement for the heatmap grid within the subplot
    grid_height = 0.7  # Height of the grid within the subplot
    grid_width = 0.9   # Width of the grid
    
    # For each p-value, create a correlation heatmap
    for p_idx, p in enumerate(p_values):
        ax = plt.gcf().add_subplot(n_rows, max_cols, p_idx+1)
        
        # Create the correlation matrix for this p value
        corr_matrix = np.zeros((len(all_solvers), len(all_solvers)))
        for i, name1 in enumerate(all_solvers):
            for j, name2 in enumerate(all_solvers):
                if i == j:
                    corr_matrix[i, j] = 1.0  # Correlation with self is 1
                elif p in correlations and name1 in correlations[p] and name2 in correlations[p][name1]:
                    corr_matrix[i, j] = correlations[p][name1][name2]
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add title
        ax.set_title(f"p = {p}")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_solvers)))
        ax.set_yticks(np.arange(len(all_solvers)))
        
        # Make labels more readable
        short_names = [name[:5] + '..' if len(name) > 7 else name for name in all_solvers]
        ax.set_xticklabels(short_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(short_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add correlation values as text
        for i in range(len(all_solvers)):
            for j in range(len(all_solvers)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", 
                              color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate title
    
    # 5. Plot showing difference between agent rewards and nearest neighbor rewards
    plt.figure(figsize=(24, 10))
    plt.title("Algorithm vs Nearest Neighbor Reward Difference", pad=15)
    
    # Plot each agent minus nearest neighbor
    for idx, name in enumerate(all_solvers):
        if name != 'NearestNeighbor':
            color_idx = idx % len(colors)
            marker_idx = idx % len(markers)
            
            # Calculate differences
            differences = []
            confidence_intervals = []
            
            for p_idx, p in enumerate(p_values):
                agent_reward = all_agents_data[name]['avg_rewards'][p_idx]
                nn_reward = all_agents_data['NearestNeighbor']['avg_rewards'][p_idx]
                diff = agent_reward - nn_reward
                
                # Calculate confidence interval for the difference
                agent_ci = all_agents_data[name]['confidence_intervals'][p_idx]
                nn_ci = all_agents_data['NearestNeighbor']['confidence_intervals'][p_idx]
                diff_ci = np.sqrt(agent_ci**2 + nn_ci**2)
                
                differences.append(diff)
                confidence_intervals.append(diff_ci)
            
            plt.plot(p_values, differences, 
                     label=f"{name} - NN", marker=markers[marker_idx], color=colors[color_idx])
            plt.fill_between(
                p_values,
                np.array(differences) - np.array(confidence_intervals),
                np.array(differences) + np.array(confidence_intervals),
                color=colors[color_idx], alpha=0.2
            )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Reward Difference")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Now create detailed plots for each agent (like in evaluate_agent)
    for agent_idx, agent_name in enumerate(agent_names):
        plt.figure(figsize=(22, 30))
        
        # Extract data for this agent
        agent_rewards = all_agents_data[agent_name]['avg_rewards']
        agent_cis = all_agents_data[agent_name]['confidence_intervals']
        agent_stds = all_agents_data[agent_name]['std_rewards']
        
        # Extract data for comparison methods
        greedy_rewards = all_agents_data['NearestNeighbor']['avg_rewards']
        greedy_cis = all_agents_data['NearestNeighbor']['confidence_intervals']
        greedy_stds = all_agents_data['NearestNeighbor']['std_rewards']

        ci_rewards = all_agents_data['Cheapest Insertion']['avg_rewards']
        ci_cis = all_agents_data['Cheapest Insertion']['confidence_intervals']
        ci_stds = all_agents_data['Cheapest Insertion']['std_rewards']
        
        # Extract OR-Tools data
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            ortools_rewards = all_agents_data['OR-Tools']['avg_rewards']
            ortools_cis = all_agents_data['OR-Tools']['confidence_intervals']
            ortools_stds = all_agents_data['OR-Tools']['std_rewards']
        
        # Extract Ant Colony data if available
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            antcolony_rewards = all_agents_data['AntColony']['avg_rewards']
            antcolony_cis = all_agents_data['AntColony']['confidence_intervals']
            antcolony_stds = all_agents_data['AntColony']['std_rewards']
        
        # 1. Plot comparing average rewards with confidence intervals
        plt.subplot(4, 2, 1)
        plt.title(f"{agent_name} Performance with 95% Confidence Intervals", pad=15)
        
        plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
        plt.plot(p_values, greedy_rewards, label="Nearest Neighbor", marker="s", color="green")

        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.plot(p_values, antcolony_rewards, label="Ant Colony", marker="^", color="purple")
        
        plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_cis),
                    np.array(agent_rewards) + np.array(agent_cis),
                    color="blue", alpha=0.3)
        
        plt.fill_between(p_values,
                    np.array(greedy_rewards) - np.array(greedy_cis),
                    np.array(greedy_rewards) + np.array(greedy_cis),
                    color="green", alpha=0.3)
        
        plt.plot(p_values, ci_rewards, label="Cheapest Insertion", marker="x", color="cyan")
        plt.fill_between(p_values,
            np.array(ci_rewards) - np.array(ci_cis),
            np.array(ci_rewards) + np.array(ci_cis),
            color="cyan", alpha=0.3)

        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_cis),
                    np.array(ortools_rewards) + np.array(ortools_cis),
                    color="red", alpha=0.3)
                    
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.fill_between(p_values,
                        np.array(antcolony_rewards) - np.array(antcolony_cis),
                        np.array(antcolony_rewards) + np.array(antcolony_cis),
                        color="purple", alpha=0.3)
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Reward")
        plt.ylim(None, 0)  # Negative rewards with 0 at top
        plt.legend()
        plt.grid(True)
        
        # 2. Plot comparing rewards with standard deviations
        plt.subplot(4, 2, 2)
        plt.title(f"{agent_name} Performance with Standard Deviations", pad=15)
        
        plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
        plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_stds),
                    np.array(agent_rewards) + np.array(agent_stds),
                    color="blue", alpha=0.2)
        
        plt.plot(p_values, greedy_rewards, label="Nearest Neighbor", marker="s", color="green")
        plt.fill_between(p_values,
                    np.array(greedy_rewards) - np.array(greedy_stds),
                    np.array(greedy_rewards) + np.array(greedy_stds),
                    color="green", alpha=0.2)
        
        plt.plot(p_values, ci_rewards, label="Cheapest Insertion", marker="x", color="cyan")
        plt.fill_between(p_values,
            np.array(ci_rewards) - np.array(ci_stds),
            np.array(ci_rewards) + np.array(ci_stds),
            color="cyan", alpha=0.2)
        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_stds),
                    np.array(ortools_rewards) + np.array(ortools_stds),
                    color="red", alpha=0.2)
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.plot(p_values, antcolony_rewards, label="Ant Colony", marker="^", color="purple")
            plt.fill_between(p_values,
                        np.array(antcolony_rewards) - np.array(antcolony_stds),
                        np.array(antcolony_rewards) + np.array(antcolony_stds),
                        color="purple", alpha=0.2)
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Reward")
        plt.ylim(None, 0)  # Negative rewards with 0 at top
        plt.legend()    
        plt.grid(True)

        # 3. Plot showing agent vs greedy difference
        plt.subplot(4, 2, 3)
        plt.title(f"{agent_name} vs Nearest Neighbor Average Reward Difference")
        agent_vs_greedy = np.array(agent_rewards) - np.array(greedy_rewards)
        plt.plot(p_values, agent_vs_greedy, color="purple", alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward - NN Reward")
        plt.grid(True, axis='y')
        
        # 4. Plot showing percentage of agent solutions better than greedy
        plt.subplot(4, 2, 4)
        plt.title(f"{agent_name} vs Nearest Neighbor Solution Comparison")
        
        # Calculate percentage of instances where agent is better
        better_than_nn = []
        identical_to_nn = []
        worse_than_nn = []
        greedy_action_pct = []
        
        for p_idx, p in enumerate(p_values):
            agent_raw_rewards = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw_rewards = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            
            better = np.sum(np.array(agent_raw_rewards) > np.array(nn_raw_rewards))
            identical = np.sum(np.array(agent_raw_rewards) == np.array(nn_raw_rewards))
            worse = np.sum(np.array(agent_raw_rewards) < np.array(nn_raw_rewards))
            
            total = len(agent_raw_rewards)
            better_than_nn.append((better / total) * 100 if total > 0 else 0)
            identical_to_nn.append((identical / total) * 100 if total > 0 else 0)
            worse_than_nn.append((worse / total) * 100 if total > 0 else 0)
            
            # Calculate average percentage of greedy choices if available
            if agent_name in all_agents_data and 'greedy_choice_pct' in all_agents_data[agent_name]:
                greedy_action_pct.append(all_agents_data[agent_name]['greedy_choice_pct'][p_idx])
            else:
                greedy_action_pct.append(0)  # Default if not available
        
        plt.plot(p_values, better_than_nn, label="% Better Than NN", marker="^", color="green")
        plt.plot(p_values, identical_to_nn, label="% Identical Solutions", marker="o", color="purple")
        plt.plot(p_values, worse_than_nn, label="% Worse Than NN", marker="s", color="orange")
        
        # Plot percentage of greedy actions if available
        if any(greedy_action_pct):
            plt.plot(p_values, greedy_action_pct, label="% Greedy Actions", marker="*", color="red")
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        # 5. Plot showing average number of precedence constraints
        plt.subplot(4, 2, 5)
        plt.title("Average Number of Precedence Constraints")
        
        # Calculate or extract constraint counts
        if 'constraint_counts' in all_agents_data:
            constraint_counts = all_agents_data['constraint_counts']
        else:
            constraint_counts = []
            for p_idx, p in enumerate(p_values):
                # Calculate the theoretical number of constraints based on p and graph size
                graph_size = all_agents_data[agent_name]['avg_rewards'][p_idx]
                theoretical_max = graph_size * (graph_size - 1) / 2  # Maximum possible constraints
                constraint_counts.append(p * theoretical_max)
                
        plt.plot(p_values, constraint_counts, marker="o", color="brown")
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Constraint Count")
        plt.grid(True)
        
        # 6. Plot showing ratio of agent reward to greedy reward
        plt.subplot(4, 2, 6)
        plt.title(f"{agent_name}/NN Reward Ratio")
        reward_ratios = []
        
        for i in range(len(p_values)):
            # Avoid division by zero or very small numbers
            if greedy_rewards[i] != 0 and abs(greedy_rewards[i]) > 1e-6:
                ratio = agent_rewards[i] / greedy_rewards[i]
                reward_ratios.append(ratio)
            else:
                reward_ratios.append(1.0)  # Default to 1.0 if greedy reward is zero
        
        plt.plot(p_values, reward_ratios, marker="o", color="darkblue")
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward / NN Reward")
        plt.grid(True)
        
        # 7. Add correlation plot
        plt.subplot(4, 2, 7)
        plt.title(f"Correlation of {agent_name} With Other Methods")
        
        # Extract correlation data for this agent with other methods
        corr_data = {}
        for other in all_solvers:
            if other != agent_name:
                corr_data[other] = []
                for p in p_values:
                    if p in correlations and agent_name in correlations[p] and other in correlations[p][agent_name]:
                        corr_data[other].append(correlations[p][agent_name][other])
                    else:
                        corr_data[other].append(0)
        
        # Plot correlation with each other method
        for idx, (other, corrs) in enumerate(corr_data.items()):
            color_idx = idx % len(colors)
            plt.plot(p_values, corrs, label=f"{agent_name} vs {other}", marker="o", color=colors[color_idx])
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.legend()
        plt.grid(True)
        
        # 8. Box plot of agent vs greedy reward difference
        plt.subplot(4, 2, 8)
        plt.title(f"{agent_name} vs NN Reward Difference Distribution", pad=15)
        
        # Prepare data for box plots
        box_data = []
        box_labels = []
        box_positions = []
        box_colors = []
        
        pos = 0
        for p_idx, p in enumerate(p_values):
            if p not in all_agents_data[agent_name]['raw_data_per_p']:
                continue
                
            agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            if or_tools_enabled:
                ort_raw = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
            # NN differences
            nn_diffs = np.array(agent_raw) - np.array(nn_raw)
            box_data[f'p{p}_vs_nn'] = nn_diffs.tolist()
            
            # OR-Tools differences
            if or_tools_enabled:
                min_len = min(len(agent_raw), len(ort_raw))
                if min_len > 0:
                    ort_diffs = np.array(agent_raw[:min_len]) - np.array(ort_raw[:min_len])
                    box_data[f'p{p}_vs_ort'] = ort_diffs.tolist()
        
        # Export as long format for easier analysis
        boxplot_rows = []
        for p in p_values:
            if f'p{p}_vs_nn' in box_data:
                for diff in box_data[f'p{p}_vs_nn']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_NN',
                        'reward_difference': diff
                    })
            
            if f'p{p}_vs_ort' in box_data:
                for diff in box_data[f'p{p}_vs_ort']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_OR_Tools',
                        'reward_difference': diff
                    })
        
        boxplot_df = pd.DataFrame(boxplot_rows)
        boxplot_df.to_csv(f"{agent_name}_boxplot_data_{instance_type}_{timestamp}.csv", index=False)
        print(f"{agent_name} boxplot data exported to {agent_name}_boxplot_data_{instance_type}_{timestamp}.csv")
    
    # Now call the plotting function as before
    create_comparison_plots(
        all_agents_data=all_agents_data,
        p_values=p_values,
        agent_names=agent_names,
        ant_solver_enabled=ant_solver_enabled,
        or_solver_enabled=or_tools_enabled,
        instance_type=instance_type,
        n_instances=n_instances,
        correlations=correlations
    )
    
    return all_agents_data, correlations, agent_names


# Add this function to create the plots at the end of evaluate_multiple_agents
def create_comparison_plots(all_agents_data, p_values, agent_names, ant_solver_enabled,or_solver_enabled, instance_type, n_instances, correlations):
    """Create comprehensive comparison plots for all solvers."""
    # Add OR-Tools and AntColony to the list of solvers to plot
    all_solvers = agent_names + ['NearestNeighbor', 'Cheapest Insertion']
    if or_solver_enabled:
        all_solvers.append('OR-Tools')
    if ant_solver_enabled:
        all_solvers.append('AntColony')
        
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'p', 'h', 'd']
    
    # First create the main comparison plots across all algorithms
    plt.figure(figsize=(24, 20))
    
    # 1. Plot comparing average rewards with confidence intervals
    plt.subplot(4, 1, 1)
    plt.title(f"Algorithm Performance Comparison ({instance_type}, {len(p_values)} p-values, {n_instances} runs)", pad=15)
    
    # Plot each solver
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['avg_rewards'],
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
        plt.fill_between(
            p_values,
            np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['confidence_intervals']),
            np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['confidence_intervals']),
            color=colors[color_idx], alpha=0.2
        )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)
    
    # 2. Plot comparing rewards with standard deviations
    plt.subplot(4, 1, 2)
    plt.title("Algorithm Performance with Standard Deviations", pad=15)
    
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['avg_rewards'], 
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
        plt.fill_between(
            p_values,
            np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['std_rewards']),
            np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['std_rewards']),
            color=colors[color_idx], alpha=0.2
        )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Average Reward")
    plt.ylim(None, 0)  # Negative rewards with 0 at top
    plt.legend()
    plt.grid(True)
    
    # 3. Plot showing percentage of best solutions
    plt.subplot(4, 1, 3)
    plt.title("Percentage of Best Solutions by Algorithm", pad=15)
    
    for idx, name in enumerate(all_solvers):
        color_idx = idx % len(colors)
        marker_idx = idx % len(markers)
        
        plt.plot(p_values, all_agents_data[name]['best_solution_pct'], 
                 label=name, marker=markers[marker_idx], color=colors[color_idx])
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Best Solution Percentage")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    # 4. Heatmap of pairwise correlations at different p values
    plt.subplot(4, 1, 4)
    plt.title("Algorithm Correlation Heatmap", pad=15)
    
    # Create a subplot for each p value
    n_p_values = len(p_values)
    max_cols = min(4, n_p_values)  # Maximum 4 columns
    n_rows = (n_p_values + max_cols - 1) // max_cols
    
    # Calculate the best placement for the heatmap grid within the subplot
    grid_height = 0.7  # Height of the grid within the subplot
    grid_width = 0.9   # Width of the grid
    
    # For each p-value, create a correlation heatmap
    for p_idx, p in enumerate(p_values):
        ax = plt.gcf().add_subplot(n_rows, max_cols, p_idx+1)
        
        # Create the correlation matrix for this p value
        corr_matrix = np.zeros((len(all_solvers), len(all_solvers)))
        for i, name1 in enumerate(all_solvers):
            for j, name2 in enumerate(all_solvers):
                if i == j:
                    corr_matrix[i, j] = 1.0  # Correlation with self is 1
                elif p in correlations and name1 in correlations[p] and name2 in correlations[p][name1]:
                    corr_matrix[i, j] = correlations[p][name1][name2]
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add title
        ax.set_title(f"p = {p}")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_solvers)))
        ax.set_yticks(np.arange(len(all_solvers)))
        
        # Make labels more readable
        short_names = [name[:5] + '..' if len(name) > 7 else name for name in all_solvers]
        ax.set_xticklabels(short_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(short_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add correlation values as text
        for i in range(len(all_solvers)):
            for j in range(len(all_solvers)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", 
                              color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate title
    
    # 5. Plot showing difference between agent rewards and nearest neighbor rewards
    plt.figure(figsize=(24, 10))
    plt.title("Algorithm vs Nearest Neighbor Reward Difference", pad=15)
    
    # Plot each agent minus nearest neighbor
    for idx, name in enumerate(all_solvers):
        if name != 'NearestNeighbor':
            color_idx = idx % len(colors)
            marker_idx = idx % len(markers)
            
            # Calculate differences
            differences = []
            confidence_intervals = []
            
            for p_idx, p in enumerate(p_values):
                agent_reward = all_agents_data[name]['avg_rewards'][p_idx]
                nn_reward = all_agents_data['NearestNeighbor']['avg_rewards'][p_idx]
                diff = agent_reward - nn_reward
                
                # Calculate confidence interval for the difference
                agent_ci = all_agents_data[name]['confidence_intervals'][p_idx]
                nn_ci = all_agents_data['NearestNeighbor']['confidence_intervals'][p_idx]
                diff_ci = np.sqrt(agent_ci**2 + nn_ci**2)
                
                differences.append(diff)
                confidence_intervals.append(diff_ci)
            
            plt.plot(p_values, differences, 
                     label=f"{name} - NN", marker=markers[marker_idx], color=colors[color_idx])
            plt.fill_between(
                p_values,
                np.array(differences) - np.array(confidence_intervals),
                np.array(differences) + np.array(confidence_intervals),
                color=colors[color_idx], alpha=0.2
            )
    
    plt.xlabel("p (Precedence Constraint Probability)")
    plt.ylabel("Reward Difference")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Now create detailed plots for each agent (like in evaluate_agent)
    for agent_idx, agent_name in enumerate(agent_names):
        plt.figure(figsize=(22, 30))
        
        # Extract data for this agent
        agent_rewards = all_agents_data[agent_name]['avg_rewards']
        agent_cis = all_agents_data[agent_name]['confidence_intervals']
        agent_stds = all_agents_data[agent_name]['std_rewards']
        
        # Extract data for comparison methods
        greedy_rewards = all_agents_data['NearestNeighbor']['avg_rewards']
        greedy_cis = all_agents_data['NearestNeighbor']['confidence_intervals']
        greedy_stds = all_agents_data['NearestNeighbor']['std_rewards']

        ci_rewards = all_agents_data['Cheapest Insertion']['avg_rewards']
        ci_cis = all_agents_data['Cheapest Insertion']['confidence_intervals']
        ci_stds = all_agents_data['Cheapest Insertion']['std_rewards']
        
        # Extract OR-Tools data
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            ortools_rewards = all_agents_data['OR-Tools']['avg_rewards']
            ortools_cis = all_agents_data['OR-Tools']['confidence_intervals']
            ortools_stds = all_agents_data['OR-Tools']['std_rewards']
        
        # Extract Ant Colony data if available
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            antcolony_rewards = all_agents_data['AntColony']['avg_rewards']
            antcolony_cis = all_agents_data['AntColony']['confidence_intervals']
            antcolony_stds = all_agents_data['AntColony']['std_rewards']
        
        # 1. Plot comparing average rewards with confidence intervals
        plt.subplot(4, 2, 1)
        plt.title(f"{agent_name} Performance with 95% Confidence Intervals", pad=15)
        
        plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
        plt.plot(p_values, greedy_rewards, label="Nearest Neighbor", marker="s", color="green")

        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.plot(p_values, antcolony_rewards, label="Ant Colony", marker="^", color="purple")
        
        plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_cis),
                    np.array(agent_rewards) + np.array(agent_cis),
                    color="blue", alpha=0.3)
        
        plt.fill_between(p_values,
                    np.array(greedy_rewards) - np.array(greedy_cis),
                    np.array(greedy_rewards) + np.array(greedy_cis),
                    color="green", alpha=0.3)
        
        plt.plot(p_values, ci_rewards, label="Cheapest Insertion", marker="x", color="cyan")
        plt.fill_between(p_values,
            np.array(ci_rewards) - np.array(ci_cis),
            np.array(ci_rewards) + np.array(ci_cis),
            color="cyan", alpha=0.3)

        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_cis),
                    np.array(ortools_rewards) + np.array(ortools_cis),
                    color="red", alpha=0.3)
                    
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.fill_between(p_values,
                        np.array(antcolony_rewards) - np.array(antcolony_cis),
                        np.array(antcolony_rewards) + np.array(antcolony_cis),
                        color="purple", alpha=0.3)
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Reward")
        plt.ylim(None, 0)  # Negative rewards with 0 at top
        plt.legend()
        plt.grid(True)
        
        # 2. Plot comparing rewards with standard deviations
        plt.subplot(4, 2, 2)
        plt.title(f"{agent_name} Performance with Standard Deviations", pad=15)
        
        plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
        plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_stds),
                    np.array(agent_rewards) + np.array(agent_stds),
                    color="blue", alpha=0.2)
        
        plt.plot(p_values, greedy_rewards, label="Nearest Neighbor", marker="s", color="green")
        plt.fill_between(p_values,
                    np.array(greedy_rewards) - np.array(greedy_stds),
                    np.array(greedy_rewards) + np.array(greedy_stds),
                    color="green", alpha=0.2)
        
        plt.plot(p_values, ci_rewards, label="Cheapest Insertion", marker="x", color="cyan")
        plt.fill_between(p_values,
            np.array(ci_rewards) - np.array(ci_stds),
            np.array(ci_rewards) + np.array(ci_stds),
            color="cyan", alpha=0.2)
        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_stds),
                    np.array(ortools_rewards) + np.array(ortools_stds),
                    color="red", alpha=0.2)
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            plt.plot(p_values, antcolony_rewards, label="Ant Colony", marker="^", color="purple")
            plt.fill_between(p_values,
                        np.array(antcolony_rewards) - np.array(antcolony_stds),
                        np.array(antcolony_rewards) + np.array(antcolony_stds),
                        color="purple", alpha=0.2)
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Reward")
        plt.ylim(None, 0)  # Negative rewards with 0 at top
        plt.legend()    
        plt.grid(True)

        # 3. Plot showing agent vs greedy difference
        plt.subplot(4, 2, 3)
        plt.title(f"{agent_name} vs Nearest Neighbor Average Reward Difference")
        agent_vs_greedy = np.array(agent_rewards) - np.array(greedy_rewards)
        plt.plot(p_values, agent_vs_greedy, color="purple", alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward - NN Reward")
        plt.grid(True, axis='y')
        
        # 4. Plot showing percentage of agent solutions better than greedy
        plt.subplot(4, 2, 4)
        plt.title(f"{agent_name} vs Nearest Neighbor Solution Comparison")
        
        # Calculate percentage of instances where agent is better
        better_than_nn = []
        identical_to_nn = []
        worse_than_nn = []
        greedy_action_pct = []
        
        for p_idx, p in enumerate(p_values):
            agent_raw_rewards = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw_rewards = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            
            better = np.sum(np.array(agent_raw_rewards) > np.array(nn_raw_rewards))
            identical = np.sum(np.array(agent_raw_rewards) == np.array(nn_raw_rewards))
            worse = np.sum(np.array(agent_raw_rewards) < np.array(nn_raw_rewards))
            
            total = len(agent_raw_rewards)
            better_than_nn.append((better / total) * 100 if total > 0 else 0)
            identical_to_nn.append((identical / total) * 100 if total > 0 else 0)
            worse_than_nn.append((worse / total) * 100 if total > 0 else 0)
            
            # Calculate average percentage of greedy choices if available
            if agent_name in all_agents_data and 'greedy_choice_pct' in all_agents_data[agent_name]:
                greedy_action_pct.append(all_agents_data[agent_name]['greedy_choice_pct'][p_idx])
            else:
                greedy_action_pct.append(0)  # Default if not available
        
        plt.plot(p_values, better_than_nn, label="% Better Than NN", marker="^", color="green")
        plt.plot(p_values, identical_to_nn, label="% Identical Solutions", marker="o", color="purple")
        plt.plot(p_values, worse_than_nn, label="% Worse Than NN", marker="s", color="orange")
        
        # Plot percentage of greedy actions if available
        if any(greedy_action_pct):
            plt.plot(p_values, greedy_action_pct, label="% Greedy Actions", marker="*", color="red")
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        # 5. Plot showing average number of precedence constraints
        plt.subplot(4, 2, 5)
        plt.title("Average Number of Precedence Constraints")
        
        # Calculate or extract constraint counts
        if 'constraint_counts' in all_agents_data:
            constraint_counts = all_agents_data['constraint_counts']
        else:
            constraint_counts = []
            for p_idx, p in enumerate(p_values):
                # Calculate the theoretical number of constraints based on p and graph size
                graph_size = all_agents_data[agent_name]['avg_rewards'][p_idx]
                theoretical_max = graph_size * (graph_size - 1) / 2  # Maximum possible constraints
                constraint_counts.append(p * theoretical_max)
                
        plt.plot(p_values, constraint_counts, marker="o", color="brown")
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Average Constraint Count")
        plt.grid(True)
        
        # 6. Plot showing ratio of agent reward to greedy reward
        plt.subplot(4, 2, 6)
        plt.title(f"{agent_name}/NN Reward Ratio")
        reward_ratios = []
        
        for i in range(len(p_values)):
            # Avoid division by zero or very small numbers
            if greedy_rewards[i] != 0 and abs(greedy_rewards[i]) > 1e-6:
                ratio = agent_rewards[i] / greedy_rewards[i]
                reward_ratios.append(ratio)
            else:
                reward_ratios.append(1.0)  # Default to 1.0 if greedy reward is zero
        
        plt.plot(p_values, reward_ratios, marker="o", color="darkblue")
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward / NN Reward")
        plt.grid(True)
        
        # 7. Add correlation plot
        plt.subplot(4, 2, 7)
        plt.title(f"Correlation of {agent_name} With Other Methods")
        
        # Extract correlation data for this agent with other methods
        corr_data = {}
        for other in all_solvers:
            if other != agent_name:
                corr_data[other] = []
                for p in p_values:
                    if p in correlations and agent_name in correlations[p] and other in correlations[p][agent_name]:
                        corr_data[other].append(correlations[p][agent_name][other])
                    else:
                        corr_data[other].append(0)
        
        # Plot correlation with each other method
        for idx, (other, corrs) in enumerate(corr_data.items()):
            color_idx = idx % len(colors)
            plt.plot(p_values, corrs, label=f"{agent_name} vs {other}", marker="o", color=colors[color_idx])
        
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.legend()
        plt.grid(True)
        
        # 8. Box plot of agent vs greedy reward difference
        plt.subplot(4, 2, 8)
        plt.title(f"{agent_name} vs NN Reward Difference Distribution", pad=15)
        
        # Prepare data for box plots
        box_data = []
        box_labels = []
        box_positions = []
        box_colors = []
        
        pos = 0
        for p_idx, p in enumerate(p_values):
            if p not in all_agents_data[agent_name]['raw_data_per_p']:
                continue
                
            agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            if or_tools_enabled:
                ort_raw = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
            # NN differences
            nn_diffs = np.array(agent_raw) - np.array(nn_raw)
            box_data[f'p{p}_vs_nn'] = nn_diffs.tolist()
            
            # OR-Tools differences
            if or_tools_enabled:
                min_len = min(len(agent_raw), len(ort_raw))
                if min_len > 0:
                    ort_diffs = np.array(agent_raw[:min_len]) - np.array(ort_raw[:min_len])
                    box_data[f'p{p}_vs_ort'] = ort_diffs.tolist()
        
        # Export as long format for easier analysis
        boxplot_rows = []
        for p in p_values:
            if f'p{p}_vs_nn' in box_data:
                for diff in box_data[f'p{p}_vs_nn']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_NN',
                        'reward_difference': diff
                    })
            
            if f'p{p}_vs_ort' in box_data:
                for diff in box_data[f'p{p}_vs_ort']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_OR_Tools',
                        'reward_difference': diff
                    })
        
        boxplot_df = pd.DataFrame(boxplot_rows)
        boxplot_df.to_csv(f"{agent_name}_boxplot_data_{instance_type}_{timestamp}.csv", index=False)
        print(f"{agent_name} boxplot data exported to {agent_name}_boxplot_data_{instance_type}_{timestamp}.csv")
    
    # Now call the plotting function as before
    create_comparison_plots(
        all_agents_data=all_agents_data,
        p_values=p_values,
        agent_names=agent_names,
        ant_solver_enabled=ant_solver_enabled,
        or_solver_enabled=or_tools_enabled,
        instance_type=instance_type,
        n_instances=n_instances,
        correlations=correlations
    )

    return all_agents_data, correlations, agent_names



"""
        
        # Calculate average rewards and confidence intervals for each agent/solver
        for name in all_agents_data.keys():
            if name in p_rewards_by_agent:
                rewards = p_rewards_by_agent[name]
                
                # Save raw data
                if p not in all_agents_data[name]['raw_data_per_p']:
                    all_agents_data[name]['raw_data_per_p'][p] = []
                all_agents_data[name]['raw_data_per_p'][p] = rewards
                
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                ci = 1.96 * np.std(rewards) / np.sqrt(len(rewards))
                best_pct = (best_agent_counts[name] / n_instances) * 100 if name in best_agent_counts else 0
                
                all_agents_data[name]['avg_rewards'].append(avg_reward)
                all_agents_data[name]["std_rewards"].append(std_reward)
                all_agents_data[name]['confidence_intervals'].append(ci)
                all_agents_data[name]['best_solution_pct'].append(best_pct)

        average_constraint_counts.append(constraint_count / n_instances)
    



    # Calculate correlations between agents
    correlations = {p: {} for p in p_values}
    
    all_solvers = list(all_agents_data.keys())
    
    for p_idx, p in enumerate(p_values):
        for name1 in all_solvers:
            correlations[p][name1] = {}
            for name2 in all_solvers:
                if name1 != name2 and name1 in all_agents_data and name2 in all_agents_data:
                    if p in all_agents_data[name1]['raw_data_per_p'] and p in all_agents_data[name2]['raw_data_per_p']:
                        rewards1 = all_agents_data[name1]['raw_data_per_p'][p]
                        rewards2 = all_agents_data[name2]['raw_data_per_p'][p]
                        
                        # Make sure arrays are the same length before correlation
                        min_len = min(len(rewards1), len(rewards2))
                        if min_len > 1:  # Need at least 2 points for correlation
                            corr, p_val = stats.pearsonr(rewards1[:min_len], rewards2[:min_len])
                        else:
                            corr, p_val = 0, 1
                        correlations[p][name1][name2] = corr

    # Export data to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    


    # 1. Export main comparison data - average rewards and confidence intervals
    comparison_data = {'p': p_values}
    
    for name in all_solvers:
        if name in all_agents_data and len(all_agents_data[name]['avg_rewards']) > 0:
            comparison_data[f'{name}_avg_reward'] = all_agents_data[name]['avg_rewards']
            comparison_data[f'{name}_std_dev'] = all_agents_data[name]['std_rewards']
            comparison_data[f'{name}_ci_lower'] = np.array(all_agents_data[name]['avg_rewards']) - np.array(all_agents_data[name]['confidence_intervals'])
            comparison_data[f'{name}_ci_upper'] = np.array(all_agents_data[name]['avg_rewards']) + np.array(all_agents_data[name]['confidence_intervals'])
            comparison_data[f'{name}_best_pct'] = all_agents_data[name]['best_solution_pct']
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"comparison_data_{instance_type}_{n_instances}runs_{timestamp}.csv", index=False)
    print(f"Comparison data exported to comparison_data_{instance_type}_{n_instances}runs_{timestamp}.csv")
    
    # 2. Export NN difference data
    nn_diff_data = {'p': p_values}
    
    for name in all_solvers:
        if name != 'NearestNeighbor':
            differences = []
            confidence_intervals = []
            
            for p_idx, p in enumerate(p_values):
                agent_reward = all_agents_data[name]['avg_rewards'][p_idx]
                nn_reward = all_agents_data['NearestNeighbor']['avg_rewards'][p_idx]
                diff = agent_reward - nn_reward
                
                agent_ci = all_agents_data[name]['confidence_intervals'][p_idx]
                nn_ci = all_agents_data['NearestNeighbor']['confidence_intervals'][p_idx]
                diff_ci = np.sqrt(agent_ci**2 + nn_ci**2)  # Simplified propagation of uncertainty
                
                differences.append(diff)
                confidence_intervals.append(diff_ci)
            
            nn_diff_data[f'{name}_diff'] = differences
            nn_diff_data[f'{name}_ci'] = confidence_intervals
    
    nn_diff_df = pd.DataFrame(nn_diff_data)
    nn_diff_df.to_csv(f"nn_differences_{instance_type}_{n_instances}runs_{timestamp}.csv", index=False)
    print(f"NN difference data exported to nn_differences_{instance_type}_{n_instances}runs_{timestamp}.csv")
    
    # 3. Export correlation data for each p value
    for p_idx, p in enumerate(p_values):
        corr_matrix = []
        
        for name1 in all_solvers:
            row = {'algorithm': name1}
            
            for name2 in all_solvers:
                if name1 == name2:
                    row[name2] = 1.0
                elif p in correlations and name1 in correlations[p] and name2 in correlations[p][name1]:
                    row[name2] = correlations[p][name1][name2]
                else:
                    row[name2] = float('nan')
            
            corr_matrix.append(row)
        
        corr_df = pd.DataFrame(corr_matrix)
        corr_df.to_csv(f"correlations_p{p}_{instance_type}_{n_instances}runs_{timestamp}.csv", index=False)
    
    print(f"Correlation matrices exported to correlations_p*_{instance_type}_{n_instances}runs_{timestamp}.csv")
    
    # 4. Export detailed data for each agent
    for agent_idx, agent_name in enumerate(agent_names):
        # Agent vs other solvers data
        agent_data = {
            'p': p_values,
            f'{agent_name}_reward': all_agents_data[agent_name]['avg_rewards'],
            f'{agent_name}_std': all_agents_data[agent_name]['std_rewards'],
            'NN_reward': all_agents_data['NearestNeighbor']['avg_rewards'],
            'NN_std': all_agents_data['NearestNeighbor']['std_rewards'],
            'CI_reward': all_agents_data['Cheapest Insertion']['avg_rewards'],
            'CI_std': all_agents_data['Cheapest Insertion']['std_rewards'],
            f'{agent_name}_vs_NN': np.array(all_agents_data[agent_name]['avg_rewards']) - np.array(all_agents_data['NearestNeighbor']['avg_rewards']),
            f'{agent_name}_vs_CI': np.array(all_agents_data[agent_name]['avg_rewards']) - np.array(all_agents_data['Cheapest Insertion']['avg_rewards']),
        }

        if or_tools_enabled and 'OR-Tools' in all_agents_data:
            agent_data['OR_Tools_reward'] = all_agents_data['OR-Tools']['avg_rewards']
            agent_data['OR_Tools_std'] = all_agents_data['OR-Tools']['std_rewards']
            agent_data[f'{agent_name}_vs_OR_Tools'] = np.array(all_agents_data[agent_name]['avg_rewards']) - np.array(all_agents_data['OR-Tools']['avg_rewards'])
        
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            agent_data['AntColony_reward'] = all_agents_data['AntColony']['avg_rewards']
            agent_data['AntColony_std'] = all_agents_data['AntColony']['std_rewards']
            agent_data[f'{agent_name}_vs_AntColony'] = np.array(all_agents_data[agent_name]['avg_rewards']) - np.array(all_agents_data['AntColony']['avg_rewards'])
        
        agent_df = pd.DataFrame(agent_data)
        agent_df.to_csv(f"{agent_name}_comparison_data_{instance_type}_{timestamp}.csv", index=False)
        print(f"{agent_name} comparison data exported to {agent_name}_comparison_data_{instance_type}_{timestamp}.csv")
        
        # Solution comparison percentages
        solution_comparison = {'p': p_values}
        
        # Calculate percentages vs NN and OR-Tools
        better_nn = []
        identical_nn = []
        worse_nn = []
        better_ort = []
        identical_ort = []
        worse_ort = []
        
        for p_idx, p in enumerate(p_values):
            # Skip if data is not available
            if p not in all_agents_data[agent_name]['raw_data_per_p']:
                continue
                
            agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            ci_raw = all_agents_data['Cheapest Insertion']['raw_data_per_p'][p]
            if or_tools_enabled:
                ort_raw = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
            # NN comparisons
            better = np.sum(np.array(agent_raw) > np.array(nn_raw))
            identical = np.sum(np.array(agent_raw) == np.array(nn_raw))
            worse = np.sum(np.array(agent_raw) < np.array(nn_raw))
            total = len(agent_raw)
            
            better_nn.append((better / total) * 100 if total > 0 else 0)
            identical_nn.append((identical / total) * 100 if total > 0 else 0)
            worse_nn.append((worse / total) * 100 if total > 0 else 0)
            
            # OR-Tools comparisons
            if or_tools_enabled:
                min_len = min(len(agent_raw), len(ort_raw))
                if min_len > 0:
                    better = np.sum(np.array(agent_raw[:min_len]) > np.array(ort_raw[:min_len]))
                    identical = np.sum(np.array(agent_raw[:min_len]) == np.array(ort_raw[:min_len]))
                    worse = np.sum(np.array(agent_raw[:min_len]) < np.array(ort_raw[:min_len]))
                
                    better_ort.append((better / min_len) * 100)
                    identical_ort.append((identical / min_len) * 100)
                    worse_ort.append((worse / min_len) * 100)
                else:
                    better_ort.append(0)
                    identical_ort.append(0)
                    worse_ort.append(0)
        
        solution_comparison['better_than_nn_pct'] = better_nn
        solution_comparison['identical_to_nn_pct'] = identical_nn
        solution_comparison['worse_than_nn_pct'] = worse_nn
        if or_tools_enabled:
            solution_comparison['better_than_ort_pct'] = better_ort
            solution_comparison['identical_to_ort_pct'] = identical_ort
            solution_comparison['worse_than_ort_pct'] = worse_ort
        
        solution_df = pd.DataFrame(solution_comparison)
        solution_df.to_csv(f"{agent_name}_solution_comparison_{instance_type}_{timestamp}.csv", index=False)
        print(f"{agent_name} solution comparison data exported to {agent_name}_solution_comparison_{instance_type}_{timestamp}.csv")
        
        # Export box plot data (raw differences)
        boxplot_data = {}
        
        for p_idx, p in enumerate(p_values):
            if p not in all_agents_data[agent_name]['raw_data_per_p']:
                continue
                
            agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            if or_tools_enabled:
                ort_raw = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
            # NN differences
            nn_diffs = np.array(agent_raw) - np.array(nn_raw)
            boxplot_data[f'p{p}_vs_nn'] = nn_diffs.tolist()
            
            # OR-Tools differences
            if or_tools_enabled:
                min_len = min(len(agent_raw), len(ort_raw))
                if min_len > 0:
                    ort_diffs = np.array(agent_raw[:min_len]) - np.array(ort_raw[:min_len])
                    boxplot_data[f'p{p}_vs_ort'] = ort_diffs.tolist()
        
        # Export as long format for easier analysis
        boxplot_rows = []
        for p in p_values:
            if f'p{p}_vs_nn' in boxplot_data:
                for diff in boxplot_data[f'p{p}_vs_nn']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_NN',
                        'reward_difference': diff
                    })
            
            if f'p{p}_vs_ort' in boxplot_data:
                for diff in boxplot_data[f'p{p}_vs_ort']:
                    boxplot_rows.append({
                        'p_value': p,
                        'comparison': 'vs_OR_Tools',
                        'reward_difference': diff
                    })
        
        boxplot_df = pd.DataFrame(boxplot_rows)
        boxplot_df.to_csv(f"{agent_name}_boxplot_data_{instance_type}_{timestamp}.csv", index=False)
        print(f"{agent_name} boxplot data exported to {agent_name}_boxplot_data_{instance_type}_{timestamp}.csv")
    
    # Now call the plotting function as before
    create_comparison_plots(
        all_agents_data=all_agents_data,
        p_values=p_values,
        agent_names=agent_names,
        ant_solver_enabled=ant_solver_enabled,
        or_solver_enabled=or_tools_enabled,
        instance_type=instance_type,
        n_instances=n_instances,
        correlations=correlations
    )
    
    return all_agents_data, correlations, agent_names

    """