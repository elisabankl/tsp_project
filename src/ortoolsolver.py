import numpy as np
from customenv import CustomEnv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

def optimal_tour_length(env):
    """Use python-tsp library to get an optimal solution."""
    distance_matrix = np.zeros((env.N_GRAPH_SIZE+1, env.N_GRAPH_SIZE+1))
    for i in range(env.N_GRAPH_SIZE):
        for j in range(env.N_GRAPH_SIZE):
            distance_matrix[i+1, j+1] = env.distance_matrix[i, j] + env.precedence_matrix[i, j]*1000
        distance_matrix[i, 1] = env.cost_matrix[i, i]
    distance_matrix[0, :] = 0
    print(distance_matrix)
    return solve_tsp_simulated_annealing(np.transpose(distance_matrix))

# Create the environment
env = CustomEnv(25)

env.reset()

# Evaluate the greedy algorithm
print("Optimal tour length: ", optimal_tour_length(env))

env.render()