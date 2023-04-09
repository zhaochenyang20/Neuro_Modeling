from modeling.neuron import Neuron
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import product

HOME = Path.cwd()
dataset_root = HOME / "dataset"
dataset_root.mkdir(parents=True, exist_ok=True)
neu_store_path = dataset_root / "neuron.npy"
if not neu_store_path.exists():
    neu = Neuron()
    np.save(str(neu_store_path), neu)
else:
    neu = np.load(str(neu_store_path), allow_pickle=True).item()

def cluster_2d(data):
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

def plot_fr_p():
    for i in neu.categories:
        neu.plot_fr_p(category=[int(i)], save_pic=True)
    neu.plot_fr_p(category=[], save_pic=True)

def plot_3d_3_field():
    for type, category in product(["fr", "p", "c"], neu.categories):
        neu.plot_self_3d(type, category=[int(category)], save_pic=True)
    for type in ["fr", "p", "c"]:
        neu.plot_self_3d(type, category=[], save_pic=True)

def cluster_3d(data, n_clusters):
    assert data.shape[1] == 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n_clusters):
        cluster_points = data[kmeans.labels_ == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    plot_3d_3_field()