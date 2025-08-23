import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt


# Import the netgen package from r
netgen = importr('netgen')

# Generate a random TSP instance
#generateRandomNetwork = ro.r('netgen::generateRandomNetwork')
generateClusteredNetwork = ro.r('netgen::generateClusteredNetwork')

tsp_instance = generateClusteredNetwork(n_points=25,n_cluster = 4, lower=0, upper=50)

# Print the TSP instance
print("Generated TSP instance:")
print(type(tsp_instance))

# Export the TSP instance to a TSPLIB file
exportToTSPlibFormat = ro.r('netgen::exportToTSPlibFormat')
#exportToTSPlibFormat(tsp_instance, file="random_tsp_instance.tsp")
#print("TSP instance exported to random_tsp_instance.tsp")

# Extract node coordinates
coordinates = np.array(ro.r("as.matrix")(tsp_instance.rx2("coordinates")))
print("Node coordinates:")
print(coordinates)

membership = np.array(tsp_instance.rx2("membership"))
print("Cluster membership for each node:")
print(membership)

# Visualize the coordinates, colored by cluster membership
plt.figure(figsize=(6, 6))
plt.scatter(
    coordinates[:, 0], coordinates[:, 1],
    c=membership, cmap='tab10', s=100, edgecolor='k'
)
plt.title("TSP Node Coordinates by Cluster")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

#will need to scramble the points because the are currently ordered by cluster, but we still need to know the cluster

print(pairwise_distances(coordinates, metric='euclidean'))