import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate some sample data
data = np.random.rand(4089, 2)

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(data)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='black')
plt.show()
