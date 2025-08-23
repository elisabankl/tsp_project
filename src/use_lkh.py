# filename: use_lkh.py
import numpy as np
from customenv import CustomEnv
from lkh_solver import LKHSolver

def calculate_solution_cost(path, distance_matrix, cost_matrix=None):
    """Calculate the total cost of a solution path."""
    total_cost = 0
    
    # Add up node costs if available
    if cost_matrix is not None:
        for node in path:
            total_cost += cost_matrix[node]
    
    # Add up edge costs
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        edge_cost = distance_matrix[from_node, to_node]
        total_cost += edge_cost
        print(f"Edge {from_node}->{to_node}: Cost = {edge_cost}")
    
    return total_cost

def main():
    # Create an environment instance
    env = CustomEnv(25, "Euclidic", p=0.1)
    
    # Create the solver
    solver = LKHSolver(timeout=30)  
    
    # Get matrices from the environment
    distance_matrix = env.distance_matrix 
    precedence_matrix = env.precedence_matrix
    
    # Unnormalize distance matrix if needed
    if hasattr(env, 'max_value'):
        distance_matrix = distance_matrix * env.max_value
    
    # Solve using LKH
    print("Solving with LKH-3...")
    solution = solver.solve_from_matrices(distance_matrix, precedence_matrix, 
                                        env.original_cost_matrix if hasattr(env, 'original_cost_matrix') else None)
    
    # Process and display the solution
    if solution:
        print("\nSolution found!")
        print("Best tour:", solution["best_solution"])
        print(f"Best tour cost: {solution['best_solution_cost']}")
        
        # Convert 1-indexed solution back to 0-indexed for visualization
        path = [x-1 for x in solution["best_solution"]]
        
        # Calculate the solution cost ourselves
        total_cost = calculate_solution_cost(path, distance_matrix, 
                                           env.original_cost_matrix if hasattr(env, 'original_cost_matrix') else None)
        print(f"Manually calculated cost: {total_cost}")
        
        # Visual confirmation of the path
        print("Path:", " -> ".join(map(str, path)))
        
    else:
        print("Failed to find a solution")

if __name__ == "__main__":
    main()