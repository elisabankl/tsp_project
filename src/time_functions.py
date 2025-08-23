import time
import numpy as np
from collections import defaultdict
from customenv import CustomEnv
import inspect

# Dictionary to store timing results
timing_stats = defaultdict(list)

# Create timing wrapper
def timing_wrapper(func, func_name):
    def wrapped(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        timing_stats[func_name].append((end_time - start_time) * 1000)  # ms
        return result
    return wrapped

# Create environment
env = CustomEnv(25, "Random")

# Save references to all methods we want to profile
original_step = env.step
original_get_obs = env._get_observation
original_update_actions = env._set_allowed_actions
original_check_constraints = env._check_precedence_constraints
original_reset = env.reset

# Also profile these additional methods that might be called during reset
if hasattr(env, "_generate_matrices"):
    original_generate_matrices = env._generate_matrices
    env._generate_matrices = timing_wrapper(original_generate_matrices, "generate_matrices")

# Try to find matrix generation functions from imported modules
from matrices import generate_random_matrices, generate_precedence_matrix, generate_cost_matrix
from matrices import generate_clustered_matrices, generate_euclidic_matrices, generate_distance_matrix

# Profile matrix generation functions
original_random_matrices = generate_random_matrices
original_precedence_matrix = generate_precedence_matrix
original_cost_matrix = generate_cost_matrix
original_clustered_matrices = generate_clustered_matrices
original_euclidic_matrices = generate_euclidic_matrices
original_distance_matrix = generate_distance_matrix


import matrices
matrices.generate_random_matrices = timing_wrapper(original_random_matrices, "generate_random_matrices")
matrices.generate_precedence_matrix = timing_wrapper(original_precedence_matrix, "generate_precedence_matrix")
matrices.generate_cost_matrix = timing_wrapper(original_cost_matrix, "generate_cost_matrix")
matrices.generate_clustered_matrices = timing_wrapper(original_clustered_matrices, "generate_clustered_matrices")
matrices.generate_euclidic_matrices = timing_wrapper(original_euclidic_matrices, "generate_euclidic_matrices")
matrices.generate_distance_matrix = timing_wrapper(original_distance_matrix, "generate_distance_matrix")

# Replace original methods with timed versions
env.step = timing_wrapper(original_step, "step")
env._get_observation = timing_wrapper(original_get_obs, "get_observation")
env._set_allowed_actions = timing_wrapper(original_update_actions, "set_allowed_actions")
env._check_precedence_constraints = timing_wrapper(original_check_constraints, "check_constraints")
env.reset = timing_wrapper(original_reset, "reset")

# Run the test with more iterations for better statistics
print("Running performance test...")
env.reset()  # Initial reset to get everything set up
steps_completed = 0
resets_needed = 0

# Clear initial timing data to get cleaner results
timing_stats.clear()
terminated = False

for _ in range(100000):  # More iterations
    allowed_actions = np.where(env.allowed_actions)[0]
    if terminated:
        env.reset()
        terminated = False
        continue
    
    action = np.random.choice(allowed_actions)
    _,_,termninated, _,_ = env.step(action)
    

# Print aggregated statistics
print(f"\nPerformance Summary ({steps_completed} steps, {resets_needed} resets):")
print("-" * 80)
print(f"{'Function':<30} {'Calls':<8} {'Mean (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Total (ms)'}")
print("-" * 80)

# Sort by total time for clearer results
sorted_stats = sorted(timing_stats.items(), key=lambda x: sum(x[1]) if x[1] else 0, reverse=True)

for func_name, times in sorted_stats:
    if not times:
        continue
        
    calls = len(times)
    mean_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    total_time = np.sum(times)
    
    print(f"{func_name:<30} {calls:<8d} {mean_time:<10.3f} {min_time:<10.3f} {max_time:<10.3f} {total_time:.3f}")

# Find the hottest function
hottest = max(timing_stats.items(), key=lambda x: sum(x[1]) if x[1] else 0)
print("\nHottest function:", hottest[0], f"({sum(hottest[1]):.3f} ms total)")

# Specifically look at reset performance
print("\nReset function breakdown:")
reset_time = sum(timing_stats["reset"])
print(f"Total reset time: {reset_time:.3f} ms across {len(timing_stats['reset'])} calls")
print(f"Average reset time: {np.mean(timing_stats['reset']):.3f} ms per call")

# Check which functions are called during reset
print("\nFunction call distribution during reset():")
for func_name, times in timing_stats.items():
    # These functions are likely called during reset
    if func_name != "reset" and any(name in func_name for name in ["generate", "matrices"]):
        percentage = sum(times) / reset_time * 100 if reset_time else 0
        print(f"  {func_name}: {percentage:.1f}% of reset time")

# Check which functions are called during step
print("\nFunction call distribution during step():")
total_step_time = sum(timing_stats["step"])
for func_name, times in timing_stats.items():
    if func_name != "step" and times and not any(name in func_name for name in ["reset", "generate", "matrices"]):
        percentage = sum(times) / total_step_time * 100 if total_step_time else 0
        print(f"  {func_name}: {percentage:.1f}% of step time")