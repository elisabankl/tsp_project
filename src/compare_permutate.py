import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from customenv import CustomEnv
import pandas as pd
from datetime import datetime
import seaborn as sns

def solve_tsp_nearest_neighbor(distance_matrix, precedence_matrix=None):
    """
    Solve the TSP problem using the nearest neighbor heuristic.
    
    Args:
        distance_matrix: Matrix of distances between nodes
        precedence_matrix: Matrix of precedence constraints (optional)
        
    Returns:
        tour: The tour as a list of node indices
        total_distance: The total distance of the tour
    """
    n = distance_matrix.shape[0]
    
    # Initialize variables
    tour = []
    visited = [False] * n
    total_distance = 0
    
    # Start at node 0 if no precedence constraints
    if precedence_matrix is None:
        current_node = 0
    else:
        # With precedence constraints, find a node with no predecessors
        has_predecessors = [False] * n
        for i in range(n):
            for j in range(n):
                if precedence_matrix[j, i] == 1:  # j must come before i
                    has_predecessors[i] = True
                    
        # Find nodes with no predecessors
        start_candidates = [i for i in range(n) if not has_predecessors[i]]
        if start_candidates:
            current_node = start_candidates[0]  # Take the first one
        else:
            current_node = 0  # Default if something is wrong with precedence matrix
    
    # Add the first node to the tour
    tour.append(current_node)
    visited[current_node] = True
    
    # Main loop: add n-1 more nodes
    while len(tour) < n:
        # Find the nearest unvisited neighbor that satisfies precedence constraints
        min_dist = float('inf')
        next_node = None
        
        for candidate in range(n):
            if visited[candidate]:
                continue  # Skip visited nodes
                
            # Check precedence constraints
            if precedence_matrix is not None:
                can_visit = True
                for pred in range(n):
                    # If pred must come before candidate, check if pred has been visited
                    if precedence_matrix[pred, candidate] == 1 and not visited[pred]:
                        can_visit = False
                        break
                if not can_visit:
                    continue  # Skip this candidate if constraints aren't satisfied
            
            # Check distance
            dist = distance_matrix[current_node, candidate]
            if dist < min_dist:
                min_dist = dist
                next_node = candidate
        
        # If no valid neighbor found, the tour is incomplete (infeasible problem)
        if next_node is None:
            # Try to find any unvisited node that satisfies precedence constraints
            for candidate in range(n):
                if not visited[candidate]:
                    if precedence_matrix is None:
                        next_node = candidate
                        min_dist = distance_matrix[current_node, candidate]
                        break
                    else:
                        # Check if all precedence constraints are satisfied
                        can_visit = True
                        for pred in range(n):
                            if precedence_matrix[pred, candidate] == 1 and not visited[pred]:
                                can_visit = False
                                break
                        if can_visit:
                            next_node = candidate
                            min_dist = distance_matrix[current_node, candidate]
                            break
        
        # If still no valid neighbor found, the tour can't be completed
        if next_node is None:
            return tour, -10000  # Return incomplete tour and a large negative reward
        
        # Add the selected node to the tour
        tour.append(next_node)
        visited[next_node] = True
        total_distance += min_dist
        current_node = next_node
    
    # Add the distance back to the starting node for a complete tour
    # (uncomment this if you need a closed tour)
    # total_distance += distance_matrix[tour[-1], tour[0]]
    
    # Return the complete tour and its total distance (negative for reward)
    return tour, -total_distance

def solve_tsp_cheapest_insertion(distance_matrix, precedence_matrix=None,cost_matrix = None):
    """
    Solve the TSP problem using the cheapest insertion heuristic.
    
    Args:
        distance_matrix: Matrix of distances between nodes
        precedence_matrix: Matrix of precedence constraints (optional)
        
    Returns:
        tour: The tour as a list of node indices
        total_distance: The total distance of the tour
    """
    n = distance_matrix.shape[0]
    if cost_matrix is None:
        cost_matrix = np.diag(np.zeros(n))
    tour = []
        
    # Track visited nodes
    visited = [False] * n

    # Start with a valid node considering precedence constraints
        # Find a node with no predecessors
    has_predecessors = [False] * n  # O(n**2)
    for i in range(n):
        for j in range(n):
            if precedence_matrix[j, i] == 1:  # j must come before i
                has_predecessors[i] = True
    start_candidates = [i for i in range(n) if not has_predecessors[i]]
    if start_candidates:
        tour = [start_candidates[np.argmin(cost_matrix[start_candidates])]]  # Take the first one
    else:
        tour = [0]  # Default if something is wrong with precedence matrix

    visited[tour[0]] = True

    # Main loop: add all remaining nodes
    while len(tour) < n:
        best_node = -1
        best_position = -1
        best_increase = float('inf')
        
        # Try each unvisited node
        for node in range(n):
            if visited[node]:
                continue  # Skip visited nodes
            
            # Check precedence constraints
            if precedence_matrix is not None:
                can_visit = True
                last_pred_in_tour = -1
                for pred in range(n):
                    if precedence_matrix[pred, node] == 1 and not visited[pred]:
                        can_visit = False
                        break
                    if precedence_matrix[pred, node] == 1 and pred in tour:
                        last_pred_in_tour = max(last_pred_in_tour,tour.index(pred))
                if not can_visit:
                    continue  # Skip if precedence constraints aren't satisfied

            
            # Try inserting at each position in the tour
            for pos in range(last_pred_in_tour+1,len(tour) + 1):
                # For insertion at the beginning
                if pos == 0:
                    increase = cost_matrix[node] +distance_matrix[node, tour[0]] - cost_matrix[tour[0]]
                # For insertion at the end
                elif pos == len(tour):
                    increase = distance_matrix[tour[-1], node]
                # For insertion between two nodes
                else:
                    # Calculate increase in tour length
                    increase = (distance_matrix[tour[pos-1], node] + 
                               distance_matrix[node, tour[pos]] - 
                               distance_matrix[tour[pos-1], tour[pos]])
                
                if increase < best_increase:
                    best_increase = increase
                    best_node = node
                    best_position = pos
        
        # If no valid insertion found, the tour is incomplete (infeasible problem)
        if best_node == -1:
            return tour, -10000  # Return incomplete tour and a large negative reward
        
        # Insert the best node at the best position
        tour.insert(best_position, best_node)
        visited[best_node] = True
    
    # Calculate the total distance of the tour
    total_distance = cost_matrix[tour[0]]  # Start with the cost of the first node
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i+1]]
    
    # Return the tour and its total distance (negative for reward)
    return tour, -total_distance

def evaluate_permutation_stability(model_path, p_values, instance_type="Random", graph_size=25, 
                                  n_instances=100, n_permutations=20):
    """
    Evaluate the stability of agent solutions against permutations of the same instance.
    
    Args:
        model_path: Path to the trained agent model
        p_values: List of p values to evaluate
        instance_type: Type of TSP instances to use
        graph_size: Size of the graph
        n_instances: Number of instances to evaluate per p value
        n_permutations: Number of permutations per instance
    """
    # Load the model
    model = MaskablePPO.load(model_path, verbose=0)
    
    # Extract agent name from path
    agent_name = model_path.split("/")[-1].split(".")[0]
    
    # Results storage - add NI data
    results = {
        'p_values': [],
        'instance_id': [],
        'permutation_id': [],
        'reward': [],
        'is_permutation': [],
        'solution_type': []  # Add solution type
    }
    
    summary_stats = {
        'p': [],
        'mean_original_reward': [],
        'std_original_reward': [],
        'mean_within_instance_std': [],
        'mean_between_instance_std': [],
        'stability_ratio': [],  # ratio of within/between variation
        'mean_nn_reward': [],   # Add NN data
        'std_nn_reward': [],    # Add NN data
        'mean_ni_reward': [],   # Add NI data
        'std_ni_reward': [],    # Add NI data
        'mean_permutation_avg': [], # Average of permutation averages
        'std_permutation_avg': []   # Standard deviation of permutation averages
    }
    
    for p_idx, p in enumerate(p_values):
        print(f"Evaluating permutation stability for p = {p}...")
        
        # Storage for this p value
        original_rewards = []
        nn_rewards = []
        ni_rewards = []  # Add NI rewards storage
        within_instance_stds = []
        all_rewards_this_p = []
        permutation_avg_rewards = []  # Average reward per instance across permutations
        
        for instance_id in range(n_instances):
            # Create a new instance
            env = CustomEnv(graph_size, instance_type, p=p)
            env.reset(fixed_instance=False)
            
            # Save the instance state
            original_instance_state = env.save_state()
            
            # Solve with original instance ordering
            obs = env._get_observation()
            done = False
            truncated = False
            original_reward = 0
            
            while not (done or truncated):
                action_masks = env.action_masks()
                action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, done, truncated, info = env.step(action)
                original_reward += reward
            
            # Record the original solution
            original_rewards.append(original_reward)
            all_rewards_this_p.append(original_reward)
            
            # Store in results
            results['p_values'].append(p)
            results['instance_id'].append(instance_id)
            results['permutation_id'].append(0)  # 0 for original
            results['reward'].append(original_reward)
            results['is_permutation'].append(False)
            results['solution_type'].append('Original')  # Mark as original solution
            
            # Reset environment for NN solution on original instance
            env.reset(fixed_instance=True)
            env.load_state(original_instance_state)
            
            # Solve with nearest neighbor
            nn_solution, nn_reward = solve_tsp_nearest_neighbor(
                env.distance_matrix,
                env.precedence_matrix
            )
            
            # Record the NN solution
            nn_rewards.append(nn_reward)
            
            # Store NN in results
            results['p_values'].append(p)
            results['instance_id'].append(instance_id)
            results['permutation_id'].append(-1)  # -1 to distinguish NN from agent
            results['reward'].append(nn_reward)
            results['is_permutation'].append(False)
            results['solution_type'].append('NearestNeighbor')
            
            # Reset environment for NI solution on original instance
            env.reset(fixed_instance=True)
            env.load_state(original_instance_state)
            
            # Solve with cheapest insertion
            ni_solution, ni_reward = solve_tsp_cheapest_insertion(
                env.distance_matrix,
                env.precedence_matrix
            )
            
            # Record the NI solution
            ni_rewards.append(ni_reward)
            
            # Store NI in results
            results['p_values'].append(p)
            results['instance_id'].append(instance_id)
            results['permutation_id'].append(-2)  # -2 to distinguish NI from agent and NN
            results['reward'].append(ni_reward)
            results['is_permutation'].append(False)
            results['solution_type'].append('CheapestInsertion')
            
            # Create storage for this instance's permutations
            permutation_rewards = []
            
            # Generate n_permutations different permutations
            for perm_id in range(n_permutations):
                # Reset to the original instance but permute node ordering
                env.reset(fixed_instance=True)
                env.load_state(original_instance_state)
                env.reset(fixed_instance=True, shuffle=True)
                
                # Solve the permuted instance
                obs = env._get_observation()
                done = False
                truncated = False
                perm_reward = 0
                
                while not (done or truncated):
                    action_masks = env.action_masks()
                    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                    obs, reward, done, truncated, info = env.step(action)
                    perm_reward += reward
                
                # Record the permutation solution
                permutation_rewards.append(perm_reward)
                all_rewards_this_p.append(perm_reward)
                
                # Store in results
                results['p_values'].append(p)
                results['instance_id'].append(instance_id)
                results['permutation_id'].append(perm_id + 1)  # 1-based for permutations
                results['reward'].append(perm_reward)
                results['is_permutation'].append(True)
                results['solution_type'].append('Permutation')
            
            # Calculate the standard deviation within this instance's permutations
            if permutation_rewards:
                within_instance_stds.append(np.std(permutation_rewards))
                permutation_avg_rewards.append(np.mean(permutation_rewards))
            
            # Add average permutation reward to results
            if permutation_rewards:
                avg_perm_reward = np.mean(permutation_rewards)
                results['p_values'].append(p)
                results['instance_id'].append(instance_id)
                results['permutation_id'].append(999)  # Use 999 to mark average
                results['reward'].append(avg_perm_reward)
                results['is_permutation'].append(True)
                results['solution_type'].append('PermutationAvg')
        
        # Calculate summary statistics for this p value
        mean_original_reward = np.mean(original_rewards)
        std_original_reward = np.std(original_rewards)
        mean_within_instance_std = np.mean(within_instance_stds) if within_instance_stds else 0
        
        # Calculate between-instance variation
        between_instance_std = std_original_reward
        
        # Calculate NN statistics
        mean_nn_reward = np.mean(nn_rewards)
        std_nn_reward = np.std(nn_rewards)
        
        # Calculate NI statistics
        mean_ni_reward = np.mean(ni_rewards)
        std_ni_reward = np.std(ni_rewards)
        
        # Calculate permutation average statistics
        mean_permutation_avg = np.mean(permutation_avg_rewards) if permutation_avg_rewards else 0
        std_permutation_avg = np.std(permutation_avg_rewards) if permutation_avg_rewards else 0
        
        # Calculate stability ratio
        stability_ratio = mean_within_instance_std / (between_instance_std + 1e-10)
        
        # Store summary stats
        summary_stats['p'].append(p)
        summary_stats['mean_original_reward'].append(mean_original_reward)
        summary_stats['std_original_reward'].append(std_original_reward)
        summary_stats['mean_within_instance_std'].append(mean_within_instance_std)
        summary_stats['mean_between_instance_std'].append(between_instance_std)
        summary_stats['stability_ratio'].append(stability_ratio)
        summary_stats['mean_nn_reward'].append(mean_nn_reward)
        summary_stats['std_nn_reward'].append(std_nn_reward)
        summary_stats['mean_ni_reward'].append(mean_ni_reward)  # Add NI mean
        summary_stats['std_ni_reward'].append(std_ni_reward)    # Add NI std
        summary_stats['mean_permutation_avg'].append(mean_permutation_avg)
        summary_stats['std_permutation_avg'].append(std_permutation_avg)
    
    # Create DataFrame for detailed results
    df = pd.DataFrame(results)
    
    # Create DataFrame for summary statistics
    summary_df = pd.DataFrame(summary_stats)
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Plot stability ratio across p values
    plt.figure(figsize=(12, 8))
    plt.plot(summary_df['p'], summary_df['stability_ratio'], marker='o', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.title(f'Solution Stability Ratio vs. p-value ({agent_name})', fontsize=14)
    plt.xlabel('p (Precedence Constraint Probability)', fontsize=12)
    plt.ylabel('Stability Ratio (Within/Between Variation)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.annotate('Ratio < 1: More stable across permutations than across instances', 
                 xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=10)
    plt.savefig(f"{agent_name}_stability_ratio_{instance_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # 2. Plot comparison of within vs between variation
    plt.figure(figsize=(12, 8))
    plt.plot(summary_df['p'], summary_df['mean_within_instance_std'], marker='o', linewidth=2, label='Within-Instance Variation')
    plt.plot(summary_df['p'], summary_df['mean_between_instance_std'], marker='s', linewidth=2, label='Between-Instance Variation')
    plt.title(f'Within vs Between Instance Variation ({agent_name})', fontsize=14)
    plt.xlabel('p (Precedence Constraint Probability)', fontsize=12)
    plt.ylabel('Standard Deviation of Rewards', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f"{agent_name}_variation_comparison_{instance_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # 3. Enhanced boxplots for each p-value showing the distribution of rewards
    for p in p_values:
        p_data = df[df['p_values'] == p]
        
        # Create the enhanced boxplot
        plt.figure(figsize=(14, 10))
        
        # Filter data by solution types for different plots
        original_data = p_data[p_data['solution_type'] == 'Original']
        permutation_data = p_data[p_data['solution_type'] == 'Permutation']
        nn_data = p_data[p_data['solution_type'] == 'NearestNeighbor']
        ni_data = p_data[p_data['solution_type'] == 'CheapestInsertion']
        avg_perm_data = p_data[p_data['solution_type'] == 'PermutationAvg']
        
        # Create a categorical color palette
        palette = {
            'Original': 'blue',
            'Permutation': 'lightblue',
            'NearestNeighbor': 'green',
            'CheapestInsertion': 'purple',
            'PermutationAvg': 'orange'
        }
        
        # Plot permutations as boxplots
        sns.boxplot(x='instance_id', y='reward', 
           data=p_data[p_data['solution_type'] == 'Permutation'], 
           color=palette['Permutation'],
           fliersize=3,
           width=0.6,
           zorder = 1)

# Add original solution as diamond markers
        sns.scatterplot(
    x='instance_id', 
    y='reward', 
    data=original_data,
    marker='D', 
    s=100, 
    color=palette['Original'],
    label='Original',
    zorder = 10
)

# Add nearest neighbor solution as diamond markers
        sns.scatterplot(
    x='instance_id', 
    y='reward', 
    data=nn_data,
    marker='D', 
    s=100, 
    color=palette['NearestNeighbor'],
    label='NearestNeighbor',
    zorder = 10
)

# Add cheapest insertion solution as diamond markers
        sns.scatterplot(
    x='instance_id', 
    y='reward', 
    data=ni_data,
    marker='D', 
    s=100, 
    color=palette['CheapestInsertion'],
    label='CheapestInsertion',
    zorder = 10
)

# Add permutation averages as diamond markers with different shape
        if not avg_perm_data.empty:
                sns.scatterplot(
        x='instance_id', 
        y='reward', 
        data=avg_perm_data,
        marker='X', 
        s=120, 
        color='orange',
        label='Permutation Avg',
        zorder = 10
    )
        
        # Add standard deviations as text
        ax = plt.gca()
        
        # Calculate statistics for this p-value
        p_stats = summary_df[summary_df['p'] == p].iloc[0]
        
        # Add text box with standard deviations - include NI
        textstr = '\n'.join((
            f'Original Std: {p_stats["std_original_reward"]:.4f}',
            f'NN Std: {p_stats["std_nn_reward"]:.4f}',
            f'NI Std: {p_stats["std_ni_reward"]:.4f}',
            f'Perm Avg Std: {p_stats["std_permutation_avg"]:.4f}',
            f'Within-Instance Std: {p_stats["mean_within_instance_std"]:.4f}',
            f'Stability Ratio: {p_stats["stability_ratio"]:.4f}'
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
        plt.title(f'Reward Distribution by Instance (p = {p})', fontsize=14)
        plt.xlabel('Instance ID', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend(title='Solution Type')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{agent_name}_p{p}_reward_distribution_{instance_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Add a new plot comparing standard deviations across p values - include NI
    plt.figure(figsize=(12, 8))
    plt.plot(summary_df['p'], summary_df['std_original_reward'], marker='o', linewidth=2, label='Original Solution Std')
    plt.plot(summary_df['p'], summary_df['std_nn_reward'], marker='s', linewidth=2, label='NN Solution Std')
    plt.plot(summary_df['p'], summary_df['std_ni_reward'], marker='*', linewidth=2, label='NI Solution Std')
    plt.plot(summary_df['p'], summary_df['std_permutation_avg'], marker='^', linewidth=2, label='Permutation Avg Std')
    plt.plot(summary_df['p'], summary_df['mean_within_instance_std'], marker='D', linewidth=2, label='Within-Instance Std')
    
    plt.title(f'Standard Deviations Comparison ({agent_name})', fontsize=14)
    plt.xlabel('p (Precedence Constraint Probability)', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f"{agent_name}_standard_deviations_{instance_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # Add a comparison plot of mean rewards by solution type
    plt.figure(figsize=(12, 8))
    plt.plot(summary_df['p'], summary_df['mean_original_reward'], marker='o', linewidth=2, label='Original Solution')
    plt.plot(summary_df['p'], summary_df['mean_nn_reward'], marker='s', linewidth=2, label='Nearest Neighbor')
    plt.plot(summary_df['p'], summary_df['mean_ni_reward'], marker='*', linewidth=2, label='Nearest Insertion')
    plt.plot(summary_df['p'], summary_df['mean_permutation_avg'], marker='^', linewidth=2, label='Permutation Avg')
    
    plt.title(f'Mean Reward Comparison by Solution Type ({agent_name})', fontsize=14)
    plt.xlabel('p (Precedence Constraint Probability)', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f"{agent_name}_mean_rewards_{instance_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # Save results to CSV
    df.to_csv(f"{agent_name}_permutation_detailed_results_{instance_type}_{timestamp}.csv", index=False)
    summary_df.to_csv(f"{agent_name}_permutation_summary_stats_{instance_type}_{timestamp}.csv", index=False)
    
    # Print summary
    print("\nPermutation Analysis Summary:")
    print("-" * 50)
    print(summary_df)
    print("-" * 50)
    print("Lower stability ratio means solutions are more stable across permutations relative to different instances.")
    
    return df, summary_df

def main():
    """Run permutation analysis for multiple agents and instance types."""
    # List of agent model paths
    agent_paths = [
        #"25_random_masked_ppo_20250626_100225.zip",
        "25_euclidic_masked_ppo_20250623_021838.zip"
        #"25_flowshop_masked_ppo_20250628_130538.zip",
        #"25_stringdistance_masked_ppo_20250629_130536.zip"
    ]
    
    # Values of p to evaluate
    p_values = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    # Instance types to evaluate
    instance_types = [#"Random", 
                      #"Euclidic",
                    #"FlowShop",
                    #"StringDistance"
                    "ClusteredWithRandomAsymmetry"
                    ]
    
    # Run analysis for each agent on its corresponding instance type
    for i, agent_path in enumerate(agent_paths):
        instance_type = instance_types[min(i, len(instance_types)-1)]
        print(f"\nAnalyzing {agent_path} on {instance_type} instances...")
        
        # Run permutation stability analysis
        detailed_results, summary_stats = evaluate_permutation_stability(
            model_path=agent_path,
            p_values=p_values,
            instance_type=instance_type,
            graph_size=25,
            n_instances=100,
            n_permutations=100
        )
        
        print(f"Analysis complete for {agent_path}.")

if __name__ == "__main__":
    main()