import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from customenv import CustomEnv
from matrices import *  # Import all matrix generation functions
import pandas as pd
from scipy import linalg
import os

def analyze_matrices(instance_types, n_matrices=200, graph_size=25, p=0.05):
    """
    Analyze properties of distance matrices for different instance types.
    
    Args:
        instance_types: List of instance types to analyze
        n_matrices: Number of matrices to generate for each type
        graph_size: Size of the matrices
        p: Precedence constraint probability
    """
    # Create directory for results
    os.makedirs("matrix_analysis", exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'instance_type': [],
        'mean_distance': [],
        'std_distance': [],
        'min_distance': [],
        'max_distance': [],
        'symmetry_ratio': [],
        'mean_asym_norm': [],
        'mean_aanti_norm': []
    }
    
    # For storing all distance values for histograms
    all_distances = {}
    
    # For each instance type
    for instance_type in instance_types:
        print(f"Analyzing {instance_type} matrices...")
        
        # Store all distance values for this instance type
        all_distances[instance_type] = []
        
        # Store symmetry metrics
        symmetry_ratios = []
        asym_norms = []
        aanti_norms = []
        
        # Generate multiple matrices
        for i in range(n_matrices):
            if i % 20 == 0:
                print(f"  Processing matrix {i+1}/{n_matrices}")
            
            # Create environment to generate matrix
            env = CustomEnv(graph_size, instance_type, p=p)
            env.reset()
            
            # Get the distance matrix
            dist_matrix = env.distance_matrix.copy()
            
            # Store all distance values for histogram
            # Exclude diagonal elements (which are typically 0)
            distances = dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)]
            all_distances[instance_type].extend(distances.flatten())
            
            # Calculate symmetry measures
            # Symmetric part: 1/2 * (A + A^T)
            A_sym = 0.5 * (dist_matrix + dist_matrix.T)
            
            # Anti-symmetric part: 1/2 * (A - A^T)
            A_anti = 0.5 * (dist_matrix - dist_matrix.T)
            
            # Compute Euclidean (Frobenius) norms
            norm_sym = linalg.norm(A_sym, 'fro')
            norm_anti = linalg.norm(A_anti, 'fro')
            
            # Compute symmetry ratio: (|A_sym| - |A_anti|) / (|A_sym| + |A_anti|)
            if norm_sym + norm_anti > 0:
                sym_ratio = (norm_sym - norm_anti) / (norm_sym + norm_anti)
            else:
                sym_ratio = 1.0  # Perfectly symmetric (and all zeros)
            
            symmetry_ratios.append(sym_ratio)
            asym_norms.append(norm_sym)
            aanti_norms.append(norm_anti)
        
        # Calculate statistics for this instance type
        results['instance_type'].append(instance_type)
        results['mean_distance'].append(np.mean(all_distances[instance_type]))
        results['std_distance'].append(np.std(all_distances[instance_type]))
        results['min_distance'].append(np.min(all_distances[instance_type]))
        results['max_distance'].append(np.max(all_distances[instance_type]))
        results['symmetry_ratio'].append(np.mean(symmetry_ratios))
        results['mean_asym_norm'].append(np.mean(asym_norms))
        results['mean_aanti_norm'].append(np.mean(aanti_norms))
        
        # Create histogram for this instance type
        plt.figure(figsize=(10, 6))
        sns.histplot(all_distances[instance_type], bins=50, kde=True)
        plt.title(f'Distance Distribution for {instance_type} Instances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig(f"matrix_analysis/{instance_type}_distance_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create symmetry ratio histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(symmetry_ratios, bins=50, kde=True)
        plt.title(f'Symmetry Ratio Distribution for {instance_type} Instances')
        plt.xlabel('Symmetry Ratio: (|A_sym| - |A_anti|) / (|A_sym| + |A_anti|)')
        plt.ylabel('Frequency')
        plt.axvline(x=1.0, color='red', linestyle='--', label='Perfect Symmetry')
        plt.axvline(x=0.0, color='green', linestyle='--', label='Equal Sym/Anti-sym')
        plt.axvline(x=-1.0, color='blue', linestyle='--', label='Perfect Anti-symmetry')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f"matrix_analysis/{instance_type}_symmetry_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("matrix_analysis/matrix_properties_summary.csv", index=False)
    print(f"Results saved to matrix_analysis/matrix_properties_summary.csv")
    
    # Create comparative visualizations
    
    # 1. Distance distributions for all instance types
    plt.figure(figsize=(12, 8))
    for instance_type in instance_types:
        sns.kdeplot(all_distances[instance_type], label=instance_type)
    plt.title('Distance Distributions by Instance Type')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("matrix_analysis/distance_distributions_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart of symmetry ratios
    plt.figure(figsize=(12, 6))
    sns.barplot(x='instance_type', y='symmetry_ratio', data=results_df)
    plt.title('Mean Symmetry Ratio by Instance Type')
    plt.xlabel('Instance Type')
    plt.ylabel('Symmetry Ratio')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("matrix_analysis/symmetry_ratio_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plots of distance values
    plt.figure(figsize=(14, 8))
    data = []
    for instance_type in instance_types:
        for dist in all_distances[instance_type][:10000]:  # Limit for performance
            data.append({'instance_type': instance_type, 'distance': dist})
    box_df = pd.DataFrame(data)
    sns.boxplot(x='instance_type', y='distance', data=box_df)
    plt.title('Distance Value Distributions by Instance Type')
    plt.xlabel('Instance Type')
    plt.ylabel('Distance')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("matrix_analysis/distance_boxplot_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nMatrix Analysis Summary:")
    print("=" * 80)
    print(results_df.round(4).to_string(index=False))
    print("=" * 80)
    
    return results_df

def visualize_example_matrices(instance_types, graph_size=25, p=0.05):
    """
    Create visualizations of example matrices for each instance type
    """
    for instance_type in instance_types:
        print(f"Visualizing example {instance_type} matrix...")
        
        # Create environment to generate matrix
        env = CustomEnv(graph_size, instance_type, p=p)
        env.reset()
        
        # Get the distance matrix
        dist_matrix = env.distance_matrix.copy()
        
        # Calculate symmetric and anti-symmetric parts
        A_sym = 0.5 * (dist_matrix + dist_matrix.T)
        A_anti = 0.5 * (dist_matrix - dist_matrix.T)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original distance matrix
        im0 = axes[0].imshow(dist_matrix, cmap='viridis')
        axes[0].set_title(f'{instance_type} Distance Matrix')
        fig.colorbar(im0, ax=axes[0])
        
        # Symmetric part
        im1 = axes[1].imshow(A_sym, cmap='viridis')
        axes[1].set_title(f'Symmetric Part: 0.5 * (A + A^T)')
        fig.colorbar(im1, ax=axes[1])
        
        # Anti-symmetric part
        im2 = axes[2].imshow(A_anti, cmap='RdBu_r')
        axes[2].set_title(f'Anti-symmetric Part: 0.5 * (A - A^T)')
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f"matrix_analysis/{instance_type}_example_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Define instance types to analyze
    instance_types = [
        "Random", 
        "Euclidic", 
        "FlowShop", 
        "StringDistance",
        "ClusteredWithRandomAsymmetry"
    ]
    
    # Analyze matrices
    results = analyze_matrices(instance_types, n_matrices=1000, graph_size=25, p=0.05)
    
    # Visualize example matrices
    visualize_example_matrices(instance_types, graph_size=25, p=0.05)
    
    print("Analysis complete. Results and visualizations saved to matrix_analysis/ directory.")