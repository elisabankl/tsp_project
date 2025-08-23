import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl

def plot_beta_distribution():
    """Plot the beta distribution used for p-value generation."""
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Parameters from your code
    alpha = 0.5
    beta = 2
    scaling = 0.45  # The scaling factor you multiply by
    
    # Generate x values for plotting
    x = np.linspace(0, 1, 1000)
    
    # Calculate PDF values for the beta distribution
    pdf_values = stats.beta.pdf(x, alpha, beta)

    shift = 0.05  # Shift to avoid zero p-values
    
    # Calculate scaled PDF values (after applying the scaling factor)
    scaled_x = x * scaling + shift

    
    # Plot the scaled Beta distribution
    plt.subplot(1, 1, 1)
    plt.plot(scaled_x, pdf_values / scaling, 'r-', linewidth=2, 
             label=f'Beta({alpha}, {beta}) × {scaling} + {shift}')
    plt.fill_between(scaled_x, pdf_values / scaling, alpha=0.2, color='red')
    plt.title(f'Scaled Beta({alpha}, {beta}) × {scaling} Distribution', fontsize=14)
    plt.xlabel('p-value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Generate samples to show histogram
    plt.figure(figsize=(10, 10))
    n_samples = 10000
    beta_samples = np.random.beta(alpha, beta, n_samples) * scaling + shift
    
    # Histogram of sampled p-values
    plt.subplot(2, 1, 1)
    counts, bins, patches = plt.hist(beta_samples, bins=50, density=True, alpha=0.6, color='green')
    
    # Add theoretical PDF curve on top of histogram
    x_theory = np.linspace(0, scaling, 1000)
    y_theory = stats.beta.pdf(x_theory / scaling, alpha, beta) / scaling
    plt.plot(x_theory, y_theory, 'r-', linewidth=2)
    
    plt.title('Distribution of Generated p-values (n=10,000)', fontsize=14)
    plt.xlabel('p-value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a custom colormap to highlight p-value ranges
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=scaling)
    
    # Add a statistics table as text
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(beta_samples, percentiles)
    
    stats_text = "P-Value Statistics:\n\n"
    stats_text += f"Min: {np.min(beta_samples):.5f}\n"
    stats_text += f"Max: {np.max(beta_samples):.5f}\n"
    stats_text += f"Mean: {np.mean(beta_samples):.5f}\n"
    stats_text += f"Median: {np.median(beta_samples):.5f}\n"
    stats_text += f"Std Dev: {np.std(beta_samples):.5f}\n\n"
    
    stats_text += "Percentiles:\n"
    for p, val in zip(percentiles, percentile_values):
        stats_text += f"{p}%: {val:.5f}\n"
    
    # Add table of statistics to the second subplot
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0.1, 0.1, stats_text, fontsize=12, family='monospace')
    
    # Create a color bar showing the full range of p-values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', 
                         label='p-value range', pad=0.2)
    
    plt.tight_layout()
    plt.savefig('beta_distribution_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional plot showing how p-values map to constraint counts
    plt.figure(figsize=(12, 6))
    
    # For different graph sizes
    graph_sizes = [10, 25, 50, 100]
    p_values = np.linspace(0, scaling, 100)
    
    for n in graph_sizes:
        # Calculate the expected number of precedence constraints
        max_constraints = n * (n - 1) / 2  # Maximum possible constraints
        expected_constraints = max_constraints * p_values
        
        plt.plot(p_values, expected_constraints, label=f'n={n}')
    
    plt.title('Expected Number of Precedence Constraints by p-value', fontsize=14)
    plt.xlabel('p-value', fontsize=12)
    plt.ylabel('Expected Constraint Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('constraint_counts_by_p.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_beta_distribution()