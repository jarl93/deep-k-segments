# libraries
import numpy as np
from sklearn.cluster import KMeans
from helpers import plot_X2D_visualization
from helpers import pairwise_distances_segments
import torch
from sklearn.decomposition import PCA
from constants import DEVICE


def k_means(X, centers_init, n_clusters, verbose = False, plot = False):
    """
    Runs k-means given the data (numpy array) and the number of clusters to consider.
    Arguments:
        X: data (numpy array) to apply k-means.
        centers_init: (numpy array) initial centers for k-means (if any).
        n_clusters: number of clusters to consider.
        verbose: boolean variable to use verbose mode.
        plot: boolean variable to plot the clustered points or not.
    Outputs:
        centers: centroids of the n_clusters clusters.
        labels: labels of the data as a result of k-means.
    """
    if centers_init is None:
        kmeans = KMeans(n_clusters=n_clusters, init='random').fit(X)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init = centers_init).fit(X)
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    if verbose:
        for i in range(n_clusters):
            print("Cluster: {}, centroid: {}: ".format(i, centers[i]))

    return centers, labels

def k_segments(X, segments_init, n_clusters, T = 300, verbose = False):
    """
    Runs the heuristic to solve k-segments as described in the paper
    'A k-segments algorithm for Finding principal curves' by  Verbeek et al.
    Paper URL: https://www.sciencedirect.com/science/article/pii/S0167865502000326.
    Arguments:
        X: data (numpy array) to apply k-means.
        segments_init: (numpy array) initial segments for k-segments heuristic.
        n_clusters: number of clusters to consider.
        T: number of iteration to run the heuristic.
        verbose: boolean variable to use verbose mode.
    Outputs:
        s: segments that represents the clusters.
        labels: labels of the data as a result of k-segments heuristic.
    """

    if type(segments_init) != torch.Tensor:
        s_tensor = torch.from_numpy(segments_init).to(DEVICE)
    else:
        s_tensor = segments_init

    X_tensor = torch.from_numpy(X).to(DEVICE)
    dim = X.shape[1]
    for t in range(T):
        distance = pairwise_distances_segments(X_tensor, s_tensor)
        labels = torch.argmin(distance, dim=1)
        labels = labels.cpu().detach().numpy()
        s_np = s_tensor.cpu().detach().numpy()
        for i in range(n_clusters):
            idx = np.where(labels == i)
            X_i = X[idx]
            # if empty cluster we skip the step
            if X_i.shape[0] == 0:
                continue
            centroid_i = np.mean(X_i, axis=0)
            pca = PCA(n_components=1)
            pca.fit(X_i)
            v_dir = pca.components_[0]
            half_ls = 1.5 * np.sqrt(pca.explained_variance_[0])
            s_i1 = centroid_i - half_ls * v_dir
            s_i2 = centroid_i + half_ls * v_dir
            s_np[i,:dim] = s_i1
            s_np[i, dim:] = s_i2
        s =  torch.from_numpy(s_np).to(DEVICE)

    labels = labels.astype(np.int64)
    if verbose:
        for i in range(n_clusters):
            print("Cluster: {}, centroid: {}: ".format(i, s[i,:]))
    return s, labels

def visualize_kmeans(writer, X, n_clusters, title_fig, title_plot):
    """
    Function to plot the visualization of k-means.
    Arguments:
        writer: writer object for tensorboard.
        X: data (numpy array) to apply k-means.
        n_clusters: number of clusters to consider.
        title_fig: title of the figure.
        title_plot: title of the plot.
    Outputs:
        None

    """
    centers, labels = k_means(X, n_clusters = n_clusters)
    title = title_fig
    writer.add_figure(title_plot,
                      plot_X2D_visualization(X, labels, title=title, num_classes = n_clusters, cluster_centers=centers))

    return None