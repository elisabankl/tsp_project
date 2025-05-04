import numpy as np
import random
import math

def generate_distance_matrix(n):
    distance_matrix = np.random.uniform(1, 10, size=(n, n))
    np.fill_diagonal(distance_matrix, 0)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = max(distance_matrix[i][j], 1)
    #Floyd-Warshall algorithm to compute the shortest paths, this is to fulfill the triangle inequality
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]: 
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    return distance_matrix

def generate_precedence_matrix(n,p = 0.1):
    precedence_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if i != j and np.random.rand() < p: #math.exp(random.uniform(math.log(0.01),math.log(0.3))):
                precedence_matrix[i][j] = 1

    reduced_precedence_matrix = precedence_matrix.copy()

    # Compute the transitive closure of the precedence matrix
    for i in range(n-2, -1, -1):
        for j in np.nonzero(precedence_matrix[i])[0]:
            for k in range(i+1, n):
                precedence_matrix[i][k] = precedence_matrix[i][k] or precedence_matrix[j][k]

    # Generate a permutation of indices
    permutation = np.random.permutation(n)

    # Apply the permutation to both rows and columns
    precedence_matrix = precedence_matrix[permutation, :][:, permutation]
    reduced_precedence_matrix = reduced_precedence_matrix[permutation, :][:, permutation]



    return precedence_matrix, reduced_precedence_matrix

def generate_cost_matrix(n):
    cost_matrix = np.diag(np.random.uniform(1, 10, size=(n)))
    return cost_matrix

def generate_matrices(n,p = 0.05):
    distance_matrix = generate_distance_matrix(n)
    precedence_matrix = generate_precedence_matrix(n,p)
    cost_matrix = generate_cost_matrix(n)
    return distance_matrix, precedence_matrix, cost_matrix