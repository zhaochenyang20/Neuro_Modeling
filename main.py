from modeling.neuron import Neuron
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
HOME = Path.cwd()
dataset_root = HOME / "dataset"
dataset_root.mkdir(parents=True, exist_ok=True)
neu_store_path = dataset_root / "neuron.npy"
if not neu_store_path.exists():
    neu = Neuron()
    np.save(str(neu_store_path), neu)
else:
    neu = np.load(str(neu_store_path), allow_pickle=True).item()

def plot():
    neu.plot_region()

def cluster(data):
    assert data.shape[1] == 2
    # Perform clustering with KMeans
    kmeans = KMeans(n_clusters=17)
    kmeans.fit(data)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot the data points with different colors for each cluster
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='black')
    plt.show()

if __name__ == "__main__":
    pass
    # plot()
    # cluster(np.asarray([neu.x, neu.y]).transpose())