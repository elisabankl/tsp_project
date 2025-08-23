# filename: use_solver.py
import numpy as np
from customenv import CustomEnv
from sop_solver import SOPSolver

def main():
    # Create an environment instance
    env = CustomEnv(25, "Random", p=0.1)
    
    # Create the solver
    solver = SOPSolver(timeout=1)  
    
    # Get matrices from the environment
    distance_matrix = env.distance_matrix
    print("Distance matrix shape:", distance_matrix)
    precedence_matrix = env.precedence_matrix
    cost_matrix = env.cost_matrix
    
    
    # Solve using ACS
    print("Solving with Ant Colony System...")
    solution = solver.solve_from_matrices(distance_matrix, precedence_matrix,np.diag(cost_matrix))
    
    # Process and display the solution
    if solution:
        print("\nSolution found!")
        print("This is the solution", solution["runs"][0]["best_solution"])
        print(f'Reported cost: {solution["runs"][0]["best_solution_value"]}')
        print(f'Execution time: {solution["total_time_sec"]}')
        
        # Convert 1-indexed solution back to 0-indexed for visualization
        path = [x-1 for x in solution["runs"][0]["best_solution"]]
        print(path)
        # Filter out any artificial nodes (those >= N)
        #path = [node for node in path if node < env.N_GRAPH_SIZE]
        
        # Calculate the solution cost ourselves
        total_cost = calculate_solution_cost(path, distance_matrix, env.original_cost_matrix)
        print(f"Manually calculated cost: {total_cost}")
        
        # Visual confirmation of the path
        print("Path:", " -> ".join(map(str, path)))
        
        
        # You could render the environment if needed
        # env.render()
        print(env.check_tour(path[1:-1]))
    else:
        print("Failed to find a solution")

def calculate_solution_cost(path, distance_matrix, cost_matrix=None):
    """
    Calculate the total cost of a solution path.
    
    Args:
        path: List of 0-indexed nodes representing the solution
        distance_matrix: Matrix of distances between nodes
        cost_matrix: Matrix of node costs (optional)
        
    Returns:
        Total cost of the solution
    """
    total_cost = 0
    
    # Add up node costs if available
    if cost_matrix is not None:
        total_cost += cost_matrix[path[1]]
    
    # Add up edge costs

    for i in range(1,len(path) - 2):
        from_node = path[i]
        to_node = path[i + 1]
        edge_cost = distance_matrix[from_node, to_node]
        total_cost += edge_cost
        print(f"Edge {from_node}->{to_node}: Cost = {edge_cost}")
    
    return total_cost


if __name__ == "__main__":
    main()