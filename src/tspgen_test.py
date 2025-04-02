import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# Import the netgen package
netgen = importr('netgen')

# Generate a random TSP instance
generateRandomNetwork = ro.r('netgen::generateRandomNetwork')
tsp_instance = generateRandomNetwork(n_points=10, lower=0, upper=100)

# Print the TSP instance
print("Generated TSP instance:")
print(type(tsp_instance))

# Export the TSP instance to a TSPLIB file
exportToTSPlibFormat = ro.r('netgen::exportToTSPlibFormat')
exportToTSPlibFormat(tsp_instance, file="random_tsp_instance.tsp")
#print("TSP instance exported to random_tsp_instance.tsp")

# Extract node coordinates
coordinates = ro.r("as.matrix")(tsp_instance.rx2("coordinates"))
print("Node coordinates:")
print(coordinates)

print(pairwise_distances(coordinates, metric='euclidean'))