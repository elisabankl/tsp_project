import numpy as np

def generate_distance_matrix(n):
    distance_matrix = np.random.uniform(1, 10, size=(n, n))
    np.fill_diagonal(distance_matrix, 0)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = max(distance_matrix[i][j], 1)
    return distance_matrix

def generate_precedence_matrix(n):
    precedence_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if i != j and np.random.rand() < 0.2:  # 20% chance of precedence
                precedence_matrix[i][j] = 1

    # Compute the transitive closure of the precedence matrix
    for i in range(n-2, -1, -1):
        for j in np.nonzero(precedence_matrix[i])[0]:
            for k in range(i+1, n):
                precedence_matrix[i][k] = precedence_matrix[i][k] or precedence_matrix[j][k]

    # Generate a permutation of indices
    permutation = np.random.permutation(n)

    # Apply the permutation to both rows and columns
    precedence_matrix = precedence_matrix[permutation, :][:, permutation]

    return precedence_matrix

def generate_cost_matrix(n):
    cost_matrix = np.diag(np.random.uniform(1, 10, size=(n)))
    return cost_matrix

def generate_matrices(n):
    distance_matrix = generate_distance_matrix(n)
    precedence_matrix = generate_precedence_matrix(n)
    cost_matrix = generate_cost_matrix(n)
    return distance_matrix, precedence_matrix, cost_matrix