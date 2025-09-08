import numpy as np
import random
import math
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix as scipy_distance_matrix

#we want to use the netgen package from R to generate TSP instances
netgen = importr('netgen')


def generate_distance_matrix(n):
    distance_matrix = np.random.uniform(1, 10, size=(n, n))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.maximum(distance_matrix, 1)
    #Floyd-Warshall algorithm to compute the shortest paths, this is to fulfill the triangle inequality
    for k in range(n):
        np.minimum(distance_matrix,
                   distance_matrix[:,np.newaxis,k]+distance_matrix[np.newaxis,k,:],
                   out = distance_matrix
                   )
    return distance_matrix

def get_transitive_closure(matrix):
    n = matrix.shape[0]
    for k in range(1,n):
        matrix = np.logical_or(
            matrix,
            np.outer(matrix[:,k],matrix[k,:])
        ).astype(np.int8)
    return matrix

def generate_precedence_matrix(n,p = 0.1):
    if p>1:
        p = np.random.beta(0.5, 2.0) * 0.45  # Skewed toward smaller values
    precedence_matrix = np.zeros((n, n), dtype=np.int8)
    upper_indices = np.triu_indices(n,k=1)
    random_values = np.random.rand(len(upper_indices[0]))
    mask = random_values < p  # Create a mask for the upper triangle based on probability p
    precedence_matrix[upper_indices[0][mask],upper_indices[1][mask]] = 1
    reduced_precedence_matrix = precedence_matrix.copy()

    precedence_matrix = get_transitive_closure(precedence_matrix)

    #permutate the matrices
    permutation = np.random.permutation(n)
    precedence_matrix = precedence_matrix[permutation, :][:, permutation]
    reduced_precedence_matrix = reduced_precedence_matrix[permutation, :][:, permutation]
    return precedence_matrix, reduced_precedence_matrix

def generate_cost_matrix(distance_matrix,n):
    cost_diag = np.random.uniform(1, 10, size=(n))
    for i in range(n):
        violations = cost_diag[i] > cost_diag + distance_matrix[:, i]
        if np.any(violations):
            cost_diag[i] = np.min(cost_diag + distance_matrix[:, i])
    return np.diag(cost_diag)

def generate_random_matrices(n,p = 0.05):
    distance_matrix = generate_distance_matrix(n)
    precedence_matrix, reduced_precedence_matrix = generate_precedence_matrix(n,p)
    cost_matrix = generate_cost_matrix(distance_matrix,n)
    #make sure the triangle inequality holds between the distance matrix and the cost matrix
    return distance_matrix, precedence_matrix, reduced_precedence_matrix, cost_matrix

def generate_clustered_matrices(n,k = 4, p_intra = 0.3,p_inter = 0.03,tilt = 1.25):
    # Generate a random TSP instance
    generateClusteredNetwork = ro.r('netgen::generateClusteredNetwork')

    tsp_instance = generateClusteredNetwork(n_points=n,n_cluster = k, lower=0, upper=50)

    # Extract node coordinates
    coordinates = np.array(ro.r("as.matrix")(tsp_instance.rx2("coordinates")))
    #get the cluster membership
    membership = np.array(tsp_instance.rx2("membership"))

    #will need to scramble the points because the are currently ordered by cluster, but we still need to know the cluster
    permutation = np.random.permutation(n)

    coordinates = coordinates[permutation]
    membership = membership[permutation]

    # Calculate the distance matrix based on the coordinates, this is based on the idea of tilting the board around the x-axis. This means that one y direction is slower becuase it is uphill, while the other is faster because it is downhill
    #the idea is from http://archive.dimacs.rutgers.edu/Challenges/TSP/papers/ATSP1.pdf tilted drilling machine with additive norm

    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = abs(coordinates[i,0]-coordinates[j,0])+max(0,coordinates[i,1]-coordinates[j,1])*tilt -min(0,coordinates[i,1]-coordinates[j,1])/tilt


    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Ensure the distance is non-negative
            distance_matrix[i][j] = max(0, distance_matrix[i][j])

    # Generate precedence constraints based on cluster membership
    precedence_matrix = np.zeros((n, n), dtype=int)
    permutation = np.random.permutation(n)
    membership = membership[permutation]
    inverse_permutation = np.argsort(permutation)

    same_cluster = membership[:,np.newaxis] == membership[np.newaxis,:]
    rand_vals = np.random.rand(n,n)
    same_cluster_mask = same_cluster & (rand_vals < p_intra)
    diff_cluster_mask = ~same_cluster & (rand_vals < p_inter)
    precedence_matrix[(same_cluster_mask | diff_cluster_mask)] = 1
    precedence_matrix = np.triu(precedence_matrix,k=1)

    reduced_precedence_matrix = precedence_matrix.copy()

    # Compute the transitive closure of the precedence matrix
    precedence_matrix = get_transitive_closure(precedence_matrix)

    # Apply the inverse permutation to both rows and columns
    precedence_matrix = precedence_matrix[inverse_permutation, :][:, inverse_permutation]
    reduced_precedence_matrix = reduced_precedence_matrix[inverse_permutation, :][:, inverse_permutation]
    cost_matrix = generate_cost_matrix(distance_matrix,n)

    return distance_matrix, precedence_matrix, reduced_precedence_matrix, cost_matrix


def generate_no_wait_flow_shop_instance(n=25, n_jobs=15,p=0.05):
    processing_times = np.random.randint(1,10, size=(n,n_jobs))
    cumulative_times = np.cumsum(processing_times, axis=1)

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            j_finish_times = cumulative_times[j]  # When j finishes on each machine
            i_cumulative = np.concatenate(([0], cumulative_times[i, :-1]))  # When i would start on each machine
            
            # Required delays for each machine
            required_delays = j_finish_times - i_cumulative
            
        # Take the maximum delay needed
            distance_matrix[j, i] = max(0, np.max(required_delays))


    precedence_matrix, reduced_precedence_matrix = generate_precedence_matrix(n,p)

    
    cost_matrix = generate_cost_matrix(distance_matrix, n)

    return distance_matrix, precedence_matrix, reduced_precedence_matrix, cost_matrix


def generate_approximate_shortest_common_superstring_instances(n=25, string_length=20, p=0.1):
    """
    Generate shortest common superstring instances efficiently.
    Distance from A to B = len(B) - max{j+2k: prefix of B of length j matches suffix of A with at most k mismatches}
    """
    # Generate all random binary strings at once
    strings = np.random.randint(0,2, size=(n,string_length))
        
    # Calculate distance matrix
    distance_matrix = np.zeros((n, n))
    
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
                
            string_a = strings[a,:]
            string_b = strings[b,:]
            
            max_overlap = 0
            
            # Vectorized overlap calculation for all possible j values
            for j in range(1, string_length + 1):
                
                # Count mismatches using vectorized XOR operation
                overlap_score = j - 2 * np.sum(string_a[-j:] != string_b[:j])
                max_overlap = max(max_overlap, overlap_score)
            
            distance_matrix[a][b] = string_length - max_overlap
    
    # Generate other matrices
    precedence_matrix, reduced_precedence_matrix = generate_precedence_matrix(n, p)
    cost_matrix = generate_cost_matrix(distance_matrix,n)
        
    return distance_matrix, precedence_matrix, reduced_precedence_matrix, cost_matrix

def generate_euclidic_matrices(n,p=0.1):
    """
    Generate Euclidean distance matrices and precedence constraints.
    """
    # Generate random coordinates
    coordinates = np.random.uniform(0, 100, size=(n, 2))
    
    # Calculate Euclidean distance matrix
    distance_matrix = pairwise_distances(coordinates, metric='euclidean')
    
    # Generate precedence constraints
    precedence_matrix, reduced_precedence_matrix = generate_precedence_matrix(n, p)
    
    cost_matrix = generate_cost_matrix(distance_matrix,n)
    
    return distance_matrix, precedence_matrix, reduced_precedence_matrix, cost_matrix


def generate_clustered_matrices_with_random_asymmetry(n,k = 4, p = 0.3,asym_prob = 1/3):
    # Generate a random TSP instance
    generateClusteredNetwork = ro.r('netgen::generateClusteredNetwork')

    tsp_instance = generateClusteredNetwork(n_points=n,n_cluster = k, lower=0, upper=50)

    # Extract node coordinates
    coordinates = np.array(ro.r("as.matrix")(tsp_instance.rx2("coordinates")))
    #get the cluster membership
    membership = np.array(tsp_instance.rx2("membership"))

    #will need to scramble the points because the are currently ordered by cluster, but we still need to know the cluster
    permutation = np.random.permutation(n)

    coordinates = coordinates[permutation]
    membership = membership[permutation]


    distance_matrix = scipy_distance_matrix(coordinates, coordinates)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.random.rand() < asym_prob :#1/3:
                #randomly introduce asymmetry, might choose a different distribution for the factor (clipped normal)
                factor = np.random.rand()+0.5
                distance_matrix[i][j] = distance_matrix[i][j]*factor

    #Floyd-Warshall algorithm to compute the shortest paths, this is to fulfill the triangle inequality (again)
    for k in range(n):
        np.minimum(distance_matrix,
                   distance_matrix[:,np.newaxis,k]+distance_matrix[np.newaxis,k,:],
                   out = distance_matrix
                   )

    

    # Generate precedence constraints based on cluster membership
    precedence_matrix = np.zeros((n, n), dtype=int)
    permutation = np.random.permutation(n)
    membership = membership[permutation]
    inverse_permutation = np.argsort(permutation)
    if p>1:
        p = np.random.beta(1.5, 3.0) * 0.5
    same_cluster = membership[:,np.newaxis] == membership[np.newaxis,:]
    rand_vals = np.random.rand(n,n)
    same_cluster_mask = same_cluster & (rand_vals < p)
    diff_cluster_mask = ~same_cluster & (rand_vals < p*0.1)
    precedence_matrix[(same_cluster_mask | diff_cluster_mask)] = 1
    precedence_matrix = np.triu(precedence_matrix,k=1)

    reduced_precedence_matrix = precedence_matrix.copy()

    # Compute the transitive closure of the precedence matrix
    precedence_matrix = get_transitive_closure(precedence_matrix)

    # Apply the inverse permutation to both rows and columns
    precedence_matrix = precedence_matrix[inverse_permutation, :][:, inverse_permutation]
    reduced_precedence_matrix = reduced_precedence_matrix[inverse_permutation, :][:, inverse_permutation]
    cost_matrix = generate_cost_matrix(distance_matrix,n)

    return distance_matrix, precedence_matrix, reduced_precedence_matrix, cost_matrix