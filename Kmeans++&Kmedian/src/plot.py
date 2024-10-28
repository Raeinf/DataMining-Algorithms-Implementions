import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_clusters(data: pd.DataFrame, centroids: pd.DataFrame, name:str , lable:str):

    plt.figure(figsize=(10, 6))
    if data.shape[1] != 3:
        raise ValueError("Data should have exactly three columns, with the last column as the cluster code.")

    if centroids is not None:
        if centroids.shape[1] != 2:
            raise ValueError("Centroids should have exactly two columns.")

    clusters = data.iloc[:, 2].astype(int)


    unique_clusters = np.sort(clusters.unique())
    for cluster in unique_clusters:
        cluster_points = data[clusters == cluster]
        plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Cluster {cluster}')

    if centroids is not None:
        plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], c='black', marker='x', s=100, linewidths=3, label='Centroids')
 
    # Add legend, title, and labels
    plt.legend()
    plt.title(name +":"+lable)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
