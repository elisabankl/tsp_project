from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from customenv import CustomEnv

    
def print_solution_tsp(data, manager, routing, solution):
    """Print TSP solution."""
    print(f"Objective: {solution.ObjectiveValue()}")
    
    index = routing.Start(0)
    plan_output = "Route: "
    route_distance = 0
    
    while not routing.IsEnd(index):
        plan_output += f"{manager.IndexToNode(index)} -> "
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    
    plan_output += f"{manager.IndexToNode(index)}"
    print(plan_output)
    print(f"Total distance: {route_distance}")



def solve_google_or_with_greedy_solution(distance_matrix,cost_matrix, precedence_matrix,timeout,verbose=False):
    """Solve TSP with precedence constraints using an initial solution."""
    
    # Extract data from observation
    n = len(distance_matrix)
    original_distance_matrix = (distance_matrix*1000).astype(int)
    cost_diagonal = np.diag((cost_matrix*1000).astype(int))
    modified_distance_matrix = np.zeros((n+1,n+1),dtype=int)
    modified_distance_matrix[1:,1:] = original_distance_matrix
    modified_distance_matrix[0,1:] = cost_diagonal




    data = {}
    data["distance_matrix"] = modified_distance_matrix
    data["distance_matrix"]
    data["num_vehicles"] = 1
    data["depot"] = 0
    
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(n+1, data["num_vehicles"], data["depot"])
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Define cost callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add time dimension for precedence constraints
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        n * 10000,  # reduced maximum time
        True,  # start cumul to zero
        "time"
    )
    
    # Get the time dimension
    time_dimension = routing.GetDimensionOrDie("time")
    
    # Add precedence constraints
    constraint_count = 0
    for i in range(n):
        for j in range(n):
            if precedence_matrix[i][j] == 1:
                pickup_index = manager.NodeToIndex(i+1)
                delivery_index = manager.NodeToIndex(j+1)
                
                routing.solver().Add(
                    time_dimension.CumulVar(pickup_index) < 
                    time_dimension.CumulVar(delivery_index)
                )
                constraint_count += 1
        
    
    # Create initial solution
    initial_solution = create_greedy_initial_solution(modified_distance_matrix,precedence_matrix)
    if verbose:
        print(f"Initial solution: {initial_solution}")
    
    # Add these diagnostic prints to solve_google_or_with_greedy_solution
    if verbose:
        print(f"Initial solution length: {len(initial_solution)}")
        print(f"Expected nodes: {n} (including depot: {n+1})")

    # Try the assignment with more detailed error capture
    try:
        assignment = routing.ReadAssignmentFromRoutes([initial_solution], True)
    except Exception as e:
        print(f"Exception during ReadAssignmentFromRoutes: {e}")
    
    if assignment:
        if verbose:
            print("Using provided initial solution")
        # Search parameters with initial solution
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        search_parameters.local_search_metaheuristic = (
           routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = timeout

        
        # Solve starting from the initial solution
        solution = routing.SolveFromAssignmentWithParameters(assignment, search_parameters)
    else:
        if verbose:
            print("Could not use initial assignment, using default solver")
        # Fallback to normal solving
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        solution = routing.SolveWithParameters(search_parameters)
    
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index-1)
            index = solution.Value(routing.NextVar(index))

        route = route[1:]
        
    
    if solution:
        if verbose:
            print_solution_tsp(data, manager, routing, solution)
            print("Solver status:", routing.status())
        return solution, route
    else:
        if verbose:
            print("No solution found even with initial solution!")
            print("Solver status:", routing.status())
        return None, route

def create_greedy_initial_solution(distance_matrix, precedence_matrix):
    """Create a greedy initial solution respecting precedence constraints."""
    n = len(distance_matrix)
    unvisited = set(range(1, n))  # Exclude depot
    current = 0  # Start at depot
    route = []
    
    while unvisited:
        # Find valid next nodes (respecting precedence)
        valid_nodes = []
        for node in unvisited:
            # Check if all precedence constraints are satisfied
            can_visit = True
            for pred in range(1,n):
                if precedence_matrix[pred-1][node-1] == 1 and pred in unvisited:
                    can_visit = False
                    break
            if can_visit:
                valid_nodes.append(node)
        
        if not valid_nodes:
            # If no valid nodes, pick the closest one (violate precedence temporarily)
            valid_nodes = list(unvisited)
        
        # Pick the closest valid node
        closest = min(valid_nodes, key=lambda x: distance_matrix[current][x])
        route.append(closest)
        unvisited.remove(closest)
        current = closest
    return route

if __name__ == "__main__":
    env = CustomEnv(25, "Random", normalize_rewards=True, p=0.01)
    observation, _ = env.reset(fixed_instance=False)
    precedence_matrix = env.reduced_precedence_matrix
    max_value = env.max_value
    observation[:,:,0] = observation[:,:,0] * max_value #revert normalization
    observation[:,:,2] = observation[:,:,2] * max_value #revert normalization
    for i in [1,2,3,5,10,15,30,60,120]:
        solution, route = solve_google_or_with_greedy_solution(observation[:,:,0],observation[:,:,2], precedence_matrix,timeout=i)
        print(f"Solution found with timeout {i} seconds: {route}")
        print(env.check_tour(route))


