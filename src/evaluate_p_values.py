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



def evaluate_multiple_agents(agent_paths, p_values, instance_type="Random", graph_size=25, n_instances=1000, ant_solver_enabled=True, or_tools_enabled=True, n_nondeterministic_runs=10, **instance_kwargs):
    """
    Evaluate multiple agents on the same set of instances.
    
    Args:
        agent_paths: List of paths to trained agent models
        p_values: List of p values to evaluate
        instance_type: Type of TSP instances to use
        graph_size: Size of the graph
        n_instances: Number of instances to evaluate per p value
        ant_solver_enabled: Whether to include Ant Colony solver
        or_tools_enabled: Whether to include OR-Tools solver
        n_nondeterministic_runs: Number of non-deterministic runs per instance
    """
    # Load all agents
    agents = []
    agent_names = []
    for path in agent_paths:
        model = MaskablePPO.load(path, verbose=0)
        agents.append(model)
        # Extract agent name from path
        agent_name = path.split("/")[-1].split(".")[0]
        agent_names.append(agent_name)
    
    # Initialize Ant Colony solver with 1 second timeout
    if ant_solver_enabled:
        try:
            from sop_solver import SOPSolver
            sop_solver = SOPSolver(timeout=1)
            print("Ant Colony Solver (SOP-ACS) initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Ant Colony Solver: {e}")
            ant_solver_enabled = False
 
    # Data structures to store results for each agent
    all_agents_data = {}
    all_solvers = agent_names + ["NearestNeighbor", "Cheapest Insertion"]
    if or_tools_enabled:
        all_solvers.append('OR-Tools')
    if ant_solver_enabled:
        all_solvers.append('AntColony')
    for agent_idx, agent_name in enumerate(agent_names):
        # Add non-deterministic version of each agent
        nondet_name = f"{agent_name}_NonDet"
        all_solvers = all_solvers + [nondet_name]

    print(all_solvers)
        
            
    for solver_name in all_solvers:
        all_agents_data[solver_name] = {
                'avg_rewards': [],
                'std_rewards': [],
                'confidence_intervals': [],
                'best_solution_pct': [],
                'raw_data_per_p': {}
            }
    
    average_constraint_counts = []
    
    # Evaluate for each value of p
    for p in p_values:
        print(f"Evaluating for p = {p}...")
        
        # Store all rewards for each agent for this p value
        p_rewards_by_agent = {name: [] for name in all_agents_data.keys()}
        
        # Store which agent had the best solution for each instance
        best_agent_counts = {name: 0 for name in all_agents_data.keys()}
        
        greedy_choice_percentages = {agent_name: [] for agent_name in agent_names}

        constraint_count = 0
        for instance in range(n_instances):
            env = CustomEnv(graph_size, instance_type, p=p, **instance_kwargs)
            obs, _ = env.reset(fixed_instance=False)
            
            constraint_count += np.sum(env.precedence_matrix)
            # Save this instance for reuse with all agents
            env_state = env.save_state()

            instance_rewards = {}
            
            # Run OR-Tools solver
            if or_tools_enabled:
                # Prepare data for OR-Tools
                distance_matrix = env.distance_matrix
                cost_matrix = np.diag(env.original_cost_matrix)
                precedence_matrix = env.reduced_precedence_matrix
                
                try:
                    solution, route = solve_google_or_with_greedy_solution(
                        distance_matrix, 
                        cost_matrix, 
                        precedence_matrix,
                        timeout=1
                    )
                
                    # Check if the solution is valid and calculate reward
                    if route and len(route) > 0:
                        # Reset environment to compute reward with OR-Tools solution
                        env.reset(fixed_instance=False)
                        env.load_state(env_state)

                        or_tools_reward, _, _ = env.check_tour(route)  # Check if the path is valid

                        # Store the rewards
                        p_rewards_by_agent['OR-Tools'].append(or_tools_reward)
                        instance_rewards['OR-Tools'] = or_tools_reward
                    else:
                        # If no solution found, assign a large negative reward
                        or_tools_reward = -10000
                        p_rewards_by_agent['OR-Tools'].append(or_tools_reward)
                        instance_rewards['OR-Tools'] = or_tools_reward
                    
                except Exception as e:
                    print(f"Error with OR-Tools solver: {e}")
                    or_tools_reward = -10000
                    p_rewards_by_agent['OR-Tools'].append(or_tools_reward)
                    instance_rewards['OR-Tools'] = or_tools_reward
            
            # Run Ant Colony solver if enabled
            if ant_solver_enabled:
                try:
                    # Prepare data for Ant Colony solver
                    distance_matrix = env.distance_matrix
                    precedence_matrix = env.precedence_matrix
                    cost_matrix = np.diag(env.cost_matrix) if hasattr(env, 'cost_matrix') else np.zeros(graph_size)
                    
                    # Solve using Ant Colony System
                    solution = sop_solver.solve_from_matrices(
                        distance_matrix=distance_matrix,
                        precedence_matrix=precedence_matrix,
                        cost_matrix=cost_matrix,
                        instance_name=f"p{p}_inst{instance}"
                    )
                    
                    if solution and "runs" in solution and solution["runs"]:
                        # Reset environment to compute reward with Ant Colony solution
                        env.reset(fixed_instance=False)
                        env.load_state(env_state)
                        
                        # Convert 1-indexed solution back to 0-indexed
                        path = [x-1 for x in solution["runs"][0]["best_solution"]]
                        
                        # For Ant Colony, we need to handle artificial nodes
                        # Skip the first and last nodes which are artificial
                        path = path[1:-1]

                        ant_colony_reward, _, _ = env.check_tour(path)  # Check if the path is valid
                        
                        # Store the rewards
                        p_rewards_by_agent['AntColony'].append(-ant_colony_reward)
                        instance_rewards['AntColony'] = ant_colony_reward
                    else:
                        # If no solution found, assign a large negative reward
                        ant_colony_reward = -10000
                        p_rewards_by_agent['AntColony'].append(ant_colony_reward)
                        instance_rewards['AntColony'] = ant_colony_reward
                        
                except Exception as e:
                    print(f"Error with AntColony solver: {e}")
                    ant_colony_reward = -100
                    p_rewards_by_agent['AntColony'].append(ant_colony_reward)
                    instance_rewards['AntColony'] = ant_colony_reward
            
            # Continue with existing agent evaluations
            for agent_idx, agent_name in enumerate(agent_names):
                agent = agents[agent_idx]
                
                # Reset to the same instance for deterministic evaluation
                env.reset(fixed_instance=False)
                env.load_state(env_state)
                
                # Solve with this agent (deterministic)
                done = False
                truncated = False
                agent_reward = 0
                
                obs = env._get_observation()
                greedy_choices_count = 0
                total_actions = 0

                while not (done or truncated):
                    action_masks = env.action_masks()
                    action, _states = agent.predict(obs, deterministic=True, action_masks=action_masks)
                    if action == greedy_action(env):
                        greedy_choices_count += 1
                    obs, rewards, done, truncated, info = env.step(action)
                    agent_reward += rewards
                    total_actions += 1
                
                p_rewards_by_agent[agent_name].append(agent_reward)
                instance_rewards[agent_name] = agent_reward
                greedy_choice_percentages[agent_name].append(greedy_choices_count / total_actions * 100 if total_actions > 0 else 0)
                
                # Now run non-deterministic evaluation multiple times
                nondet_name = f"{agent_name}_NonDet"
                nondet_rewards = []
                
                for nondet_run in range(n_nondeterministic_runs):
                    # Reset to the same instance
                    env.reset(fixed_instance=False)
                    env.load_state(env_state)
                    
                    # Solve with this agent (non-deterministic)
                    done = False
                    truncated = False
                    nondet_reward = 0
                    
                    obs = env._get_observation()
                    
                    while not (done or truncated):
                        action_masks = env.action_masks()
                        action, _states = agent.predict(obs, deterministic=False, action_masks=action_masks)
                        obs, reward, done, truncated, info = env.step(action)
                        nondet_reward += reward
                    
                    nondet_rewards.append(nondet_reward)
                
                # Average the non-deterministic runs
                avg_nondet_reward = np.mean(nondet_rewards)
                p_rewards_by_agent[nondet_name].append(avg_nondet_reward)
                instance_rewards[nondet_name] = avg_nondet_reward

            # First get nearest neighbor solution
            env.reset(fixed_instance=False)
            env.load_state(env_state)
            env._get_observation()
            done = False
            truncated = False
            nn_reward = 0
            while not (done or truncated):
                action = greedy_action(env)
                obs, reward, done, truncated, info = env.step(action)
                nn_reward += reward
            
            p_rewards_by_agent['NearestNeighbor'].append(nn_reward)
            instance_rewards['NearestNeighbor'] = nn_reward

            # Get the cheapest insertion solution
            env.reset(fixed_instance=False)
            env.load_state(env_state)
            
            # Solve with cheapest insertion
            ni_solution, ci_reward = solve_tsp_cheapest_insertion(
                env.distance_matrix,
                env.precedence_matrix
            )

            p_rewards_by_agent['Cheapest Insertion'].append(ci_reward)
            instance_rewards['Cheapest Insertion'] = ci_reward
            
            # Determine which agent had the best solution for this instance
            best_reward = max(instance_rewards.values())
            for name, reward in instance_rewards.items():
                if reward == best_reward:  # In case of ties, all get a count
                    best_agent_counts[name] += 1
        
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
    
    # Save the main comparison plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_plot_filename = f"agent_comparison_{instance_type}_{n_instances}runs_{timestamp}.png"
    plt.savefig(main_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Main comparison plots saved as {main_plot_filename}")
    
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
            # Extract raw rewards for this p value
            agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
            nn_raw = all_agents_data['NearestNeighbor']['raw_data_per_p'][p]
            
            # Calculate differences
            diffs = np.array(agent_raw) - np.array(nn_raw)
            
            box_data.append(diffs)
            box_labels.append(f'{p}')
            box_positions.append(pos)
            box_colors.append('lightgreen' if np.mean(diffs) > 0 else 'lightcoral')
            pos += 1
            # Add spacing between p-values
            pos += 0.5
        
        # Create box plots
        bp = plt.boxplot(box_data, positions=box_positions, patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        
        plt.xticks(box_positions, box_labels)
        plt.xlabel("p (Precedence Constraint Probability)")
        plt.ylabel(f"{agent_name} Reward - NN Reward")
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Save the agent-specific plots
        agent_plot_filename = f"{agent_name}_evaluation_{instance_type}_{timestamp}.png"
        plt.savefig(agent_plot_filename, dpi=300, bbox_inches='tight')
        print(f"{agent_name} detailed plots saved as {agent_plot_filename}")
        
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            
            # Now create a new figure for OR-Tools comparison
            plt.figure(figsize=(22, 24))  # 6 plots instead of 8, so reduce height
        
            # 1. Plot comparing average rewards with confidence intervals (vs OR-Tools)
            plt.subplot(3, 2, 1)
            plt.title(f"{agent_name} vs OR-Tools Performance with 95% Confidence Intervals", pad=15)
        
            plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
        
            plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_cis),
                    np.array(agent_rewards) + np.array(agent_cis),
                    color="blue", alpha=0.3)
        
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_cis),
                    np.array(ortools_rewards) + np.array(ortools_cis),
                    color="red", alpha=0.3)
        
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel("Average Reward")
            plt.ylim(None, 0)  # Negative rewards with 0 at top
            plt.legend()
            plt.grid(True)
        
            # 2. Plot comparing rewards with standard deviations (vs OR-Tools)
            plt.subplot(3, 2, 2)
            plt.title(f"{agent_name} vs OR-Tools Performance with Standard Deviations", pad=15)
        
            plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
            plt.fill_between(p_values,
                    np.array(agent_rewards) - np.array(agent_stds),
                    np.array(agent_rewards) + np.array(agent_stds),
                    color="blue", alpha=0.2)
        
            plt.plot(p_values, ortools_rewards, label="OR-Tools", marker="D", color="red")
            plt.fill_between(p_values,
                    np.array(ortools_rewards) - np.array(ortools_stds),
                    np.array(ortools_rewards) + np.array(ortools_stds),
                    color="red", alpha=0.2)
        
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel("Average Reward")
            plt.ylim(None, 0)  # Negative rewards with 0 at top
            plt.legend()    
            plt.grid(True)
        
            # 3. Plot showing agent vs OR-Tools difference
            plt.subplot(3, 2, 3)
            plt.title(f"{agent_name} vs OR-Tools Average Reward Difference")
            agent_vs_ortools = np.array(agent_rewards) - np.array(ortools_rewards)
            plt.plot(p_values, agent_vs_ortools, color="purple", alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel(f"{agent_name} Reward - OR-Tools Reward")
            plt.grid(True, axis='y')
        
            # 4. Plot showing percentage of agent solutions better than OR-Tools
            plt.subplot(3, 2, 4)
            plt.title(f"{agent_name} vs OR-Tools Solution Comparison")
        
            # Calculate percentage of instances where agent is better
            better_than_ort = []
            identical_to_ort = []
            worse_than_ort = []
        
            for p_idx, p in enumerate(p_values):
                agent_raw_rewards = all_agents_data[agent_name]['raw_data_per_p'][p]
                ort_raw_rewards = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
                # Make sure arrays are the same length
                min_len = min(len(agent_raw_rewards), len(ort_raw_rewards))
                agent_raw_rewards = agent_raw_rewards[:min_len]
                ort_raw_rewards = ort_raw_rewards[:min_len]
            
                better = np.sum(np.array(agent_raw_rewards) > np.array(ort_raw_rewards))
                identical = np.sum(np.array(agent_raw_rewards) == np.array(ort_raw_rewards))
                worse = np.sum(np.array(agent_raw_rewards) < np.array(ort_raw_rewards))
            
                total = len(agent_raw_rewards)
                better_than_ort.append((better / total) * 100 if total > 0 else 0)
                identical_to_ort.append((identical / total) * 100 if total > 0 else 0)
                worse_than_ort.append((worse / total) * 100 if total > 0 else 0)
        
            plt.plot(p_values, better_than_ort, label="% Better Than OR-Tools", marker="^", color="green")
            plt.plot(p_values, identical_to_ort, label="% Identical Solutions", marker="o", color="purple")
            plt.plot(p_values, worse_than_ort, label="% Worse Than OR-Tools", marker="s", color="orange")
        
            plt.xlabel("p (Precedence Constraint Probability)") 
            plt.ylabel("Percentage (%)")
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True)
        
            # 5. Plot showing ratio of agent reward to OR-Tools reward
            plt.subplot(3, 2, 5)
            plt.title(f"{agent_name}/OR-Tools Reward Ratio")
            reward_ratios = []
        
            for i in range(len(p_values)):
                # Avoid division by zero or very small numbers
                if ortools_rewards[i] != 0 and abs(ortools_rewards[i]) > 1e-6:
                    ratio = agent_rewards[i] / ortools_rewards[i]
                    reward_ratios.append(ratio)
                else:
                    reward_ratios.append(1.0)  # Default to 1.0 if OR-Tools reward is zero
        
            plt.plot(p_values, reward_ratios, marker="o", color="darkblue")
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel(f"{agent_name} Reward / OR-Tools Reward")
            plt.grid(True)
        
            # 6. Box plot of agent vs OR-Tools reward difference
            plt.subplot(3, 2, 6)
            plt.title(f"{agent_name} vs OR-Tools Reward Difference Distribution", pad=15)
        
            # Prepare data for box plots
            box_data = []
            box_labels = []
            box_positions = []
            box_colors = []
        
            pos = 0
            for p_idx, p in enumerate(p_values):
                # Extract raw rewards for this p value
                agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
                ort_raw = all_agents_data['OR-Tools']['raw_data_per_p'][p]
            
                # Make sure arrays are the same length
                min_len = min(len(agent_raw), len(ort_raw))
                agent_raw = agent_raw[:min_len]
                ort_raw = ort_raw[:min_len]
            
                # Calculate differences
                diffs = np.array(agent_raw) - np.array(ort_raw)
            
                box_data.append(diffs)
                box_labels.append(f'{p}')
                box_positions.append(pos)
                box_colors.append('lightgreen' if np.mean(diffs) > 0 else 'lightcoral')
                pos += 1
            
                # Add spacing between p-values
                pos += 0.5
        
            # Create box plots
            bp = plt.boxplot(box_data, positions=box_positions, patch_artist=True, widths=0.6)
        
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
        
            plt.xticks(box_positions, box_labels)
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel(f"{agent_name} Reward - OR-Tools Reward")
        
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.grid(True, axis='y')
        
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
            # Save the OR-Tools comparison plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ortools_comparison_filename = f"{agent_name}_vs_ortools_{instance_type}_{timestamp}.png"
            plt.savefig(ortools_comparison_filename, dpi=300, bbox_inches='tight')
            print(f"{agent_name} vs OR-Tools comparison plots saved as {ortools_comparison_filename}")

        # Replace the OR-Tools specific comparison section with this general approach
    # Get the list of external solvers to compare against
        external_solvers = ["Cheapest Insertion"]
        if or_solver_enabled and 'OR-Tools' in all_agents_data:
            external_solvers.append('OR-Tools')
        if ant_solver_enabled and 'AntColony' in all_agents_data:
            external_solvers.append('AntColony')
    
        # Loop through each external solver
        for solver_name in external_solvers:
            # Create a new figure for solver comparison
            plt.figure(figsize=(22, 24))  # 6 plots instead of 8, so reduce height
        
            # Extract solver data
            solver_rewards = all_agents_data[solver_name]['avg_rewards']
            solver_cis = all_agents_data[solver_name]['confidence_intervals']
            solver_stds = all_agents_data[solver_name]['std_rewards']
        
            # 1. Plot comparing average rewards with confidence intervals (vs solver)
            plt.subplot(3, 2, 1)
            plt.title(f"{agent_name} vs {solver_name} Performance with 95% Confidence Intervals", pad=15)
        
            plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
            plt.plot(p_values, solver_rewards, label=solver_name, marker="D", color="red")
        
            plt.fill_between(p_values,
                np.array(agent_rewards) - np.array(agent_cis),
                np.array(agent_rewards) + np.array(agent_cis),
                color="blue", alpha=0.3)
        
            plt.fill_between(p_values,
                np.array(solver_rewards) - np.array(solver_cis),
                np.array(solver_rewards) + np.array(solver_cis),
                color="red", alpha=0.3)
        
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel("Average Reward")
            plt.ylim(None, 0)  # Negative rewards with 0 at top
            plt.legend()
            plt.grid(True)
        
            # 2. Plot comparing rewards with standard deviations (vs solver)
            plt.subplot(3, 2, 2)
            plt.title(f"{agent_name} vs {solver_name} Performance with Standard Deviations", pad=15)
        
            plt.plot(p_values, agent_rewards, label=agent_name, marker="o", color="blue")
            plt.fill_between(p_values,
                np.array(agent_rewards) - np.array(agent_stds),
                np.array(agent_rewards) + np.array(agent_stds),
                color="blue", alpha=0.2)
        
            plt.plot(p_values, solver_rewards, label=solver_name, marker="D", color="red")
            plt.fill_between(p_values,
                np.array(solver_rewards) - np.array(solver_stds),
                np.array(solver_rewards) + np.array(solver_stds),
                color="red", alpha=0.2)
        
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel("Average Reward")
            plt.ylim(None, 0)  # Negative rewards with 0 at top
            plt.legend()    
            plt.grid(True)
        
            # 3. Plot showing agent vs solver difference
            plt.subplot(3, 2, 3)
            plt.title(f"{agent_name} vs {solver_name} Average Reward Difference")
            agent_vs_solver = np.array(agent_rewards) - np.array(solver_rewards)
            plt.plot(p_values, agent_vs_solver, color="purple", alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel(f"{agent_name} Reward - {solver_name} Reward")
            plt.grid(True, axis='y')
        
            # 4. Plot showing percentage of agent solutions better than solver
            plt.subplot(3, 2, 4)
            plt.title(f"{agent_name} vs {solver_name} Solution Comparison")
        
            # Calculate percentage of instances where agent is better
            better_than_solver = []
            identical_to_solver = []
            worse_than_solver = []
        
            for p_idx, p in enumerate(p_values):
                agent_raw_rewards = all_agents_data[agent_name]['raw_data_per_p'][p]
                solver_raw_rewards = all_agents_data[solver_name]['raw_data_per_p'][p]
            
                # Make sure arrays are the same length
                min_len = min(len(agent_raw_rewards), len(solver_raw_rewards))
                agent_raw_rewards = agent_raw_rewards[:min_len]
                solver_raw_rewards = solver_raw_rewards[:min_len]
            
                better = np.sum(np.array(agent_raw_rewards) > np.array(solver_raw_rewards))
                identical = np.sum(np.array(agent_raw_rewards) == np.array(solver_raw_rewards))
                worse = np.sum(np.array(agent_raw_rewards) < np.array(solver_raw_rewards))
            
                total = len(agent_raw_rewards)
                better_than_solver.append((better / total) * 100 if total > 0 else 0)
                identical_to_solver.append((identical / total) * 100 if total > 0 else 0)
                worse_than_solver.append((worse / total) * 100 if total > 0 else 0)
        
            plt.plot(p_values, better_than_solver, label=f"% Better Than {solver_name}", marker="^", color="green")
            plt.plot(p_values, identical_to_solver, label="% Identical Solutions", marker="o", color="purple")
            plt.plot(p_values, worse_than_solver, label=f"% Worse Than {solver_name}", marker="s", color="orange")
        
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel("Percentage (%)")
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True)
        
            # 5. Plot showing ratio of agent reward to solver reward
            plt.subplot(3, 2, 5)
            plt.title(f"{agent_name}/{solver_name} Reward Ratio")
            reward_ratios = []
        
            for i in range(len(p_values)):
                # Avoid division by zero or very small numbers
                if solver_rewards[i] != 0 and abs(solver_rewards[i]) > 1e-6:
                    ratio = agent_rewards[i] / solver_rewards[i]
                    reward_ratios.append(ratio)
                else:
                    reward_ratios.append(1.0)  # Default to 1.0 if solver reward is zero
        
            plt.plot(p_values, reward_ratios, marker="o", color="darkblue")
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel(f"{agent_name} Reward / {solver_name} Reward")
            plt.grid(True)
        
            # 6. Box plot of agent vs solver reward difference
            plt.subplot(3, 2, 6)
            plt.title(f"{agent_name} vs {solver_name} Reward Difference Distribution", pad=15)
        
            # Prepare data for box plots
            box_data = []
            box_labels = []
            box_positions = []
            box_colors = []
        
            pos = 0
            for p_idx, p in enumerate(p_values):
                # Extract raw rewards for this p value
                agent_raw = all_agents_data[agent_name]['raw_data_per_p'][p]
                solver_raw = all_agents_data[solver_name]['raw_data_per_p'][p]
            
                # Make sure arrays are the same length
                min_len = min(len(agent_raw), len(solver_raw))
                agent_raw = agent_raw[:min_len]
                solver_raw = solver_raw[:min_len]
            
                # Calculate differences
                diffs = np.array(agent_raw) - np.array(solver_raw)
            
                box_data.append(diffs)
                box_labels.append(f'{p}')
                box_positions.append(pos)
                box_colors.append('lightgreen' if np.mean(diffs) > 0 else 'lightcoral')
                pos += 1
            
                # Add spacing between p-values
                pos += 0.5
        
            # Create box plots
            bp = plt.boxplot(box_data, positions=box_positions, patch_artist=True, widths=0.6)
        
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
        
            plt.xticks(box_positions, box_labels)
            plt.xlabel("p (Precedence Constraint Probability)")
            plt.ylabel(f"{agent_name} Reward - {solver_name} Reward")
        
            # Add a horizontal line at y=0
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.grid(True, axis='y')
        
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
            # Save the solver comparison plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            solver_comparison_filename = f"{agent_name}_vs_{solver_name.lower()}_{instance_type}_{timestamp}.png"
            plt.savefig(solver_comparison_filename, dpi=300, bbox_inches='tight')
            print(f"{agent_name} vs {solver_name} comparison plots saved as {solver_comparison_filename}")



def evaluate_agents(instance_type,agent_paths, ant_solver_enabled=True,or_solver_enabled=True, **instance_kwargs):
    """Main function to evaluate and compare multiple agents."""
    # List of agent model paths to compare
    
    
    # Values of p to evaluate
    p_values = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.125,0.15,0.2,0.3]
    
    n_instances = 100  # Reduced for testing, increase for production
    graph_size = 25
    
    # Run evaluation with all agents and solvers
    all_agents_data, correlations, agent_names = evaluate_multiple_agents(
        agent_paths=agent_paths,
        p_values=p_values,
        instance_type=instance_type,
        ant_solver_enabled=ant_solver_enabled,
        or_tools_enabled=or_solver_enabled,
        graph_size=graph_size,
        n_instances=n_instances,
        **instance_kwargs
    )
    
    print(f"Evaluation complete. Results saved to agent_comparison_{instance_type}_{n_instances}runs.csv")
    print(f"Plots saved to agent_comparison_{instance_type}_{n_instances}runs_*.png")

if __name__ == "__main__":
    # Everything is now properly contained in functions
    # Only the evaluate_agents function is called here
    agent_paths = [
        "25_flowshop_masked_ppo_20250628_130538.zip",
        "25_euclidic_masked_ppo_20250623_021838.zip",
        "25_random_masked_ppo_20250626_100225.zip",
        "25_stringdistance_masked_ppo_20250629_130536.zip",  
        "25_euclidic_masked_ppo_20250623_021838.zip" 
         ]
    instance_types = ["FlowShop","Euclidic","Random","StringDistance","ClusteredWithRandomAsymmetry"]
    for i in range(1):
        instance_type = "FlowShop" #instance_types[i]
        agent_path = ["25_flowshop_masked_ppo_20250628_130538.zip"] #,"25_clusteredwithrandomasymmetry_masked_ppo_20250805_222638.zip"]

        print(f"Evaluating agent {agent_path} for instance type: {instance_type}")
        evaluate_agents(instance_type=instance_type,agent_paths = agent_path, or_solver_enabled=False,ant_solver_enabled=False, n_jobs=5)
    #evaluate_agent("25_clusteredwithrandomasymmetry_masked_ppo_20250714_230351",instance_type="ClusteredWithRandomAsymmetry", graph_size=25)