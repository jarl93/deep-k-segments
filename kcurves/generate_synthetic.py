import os
import argparse
from helpers import transform_low_to_high, load_config, sigmoid, plot_functions, plot_X2D_visualization, \
    plot_2D_visualization_generation_functions
from data_management import normalize_data
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from functions import FunctionSin
from datetime import datetime
from clustering import k_means
from sklearn.datasets import make_blobs
from constants import DATA_SET # this constant should match with the one in scripts.constants

def generate_list_W_and_F(cfg_file):
    """
        Generates a list of dimensions to apply a linear transformation and also defines
        the non-linearity function.
        Arguments:
            cfg_file: config file with the list of dimensions for the linear transformations.
    """

    # define and store the linear transformations
    list_dimensions = []
    for dimension in cfg_file["transformation"]["list_dimensions"]:
        list_dimensions.append(tuple(dimension))

    num_transformations = len(list_dimensions)
    list_W = []

    for i in range(num_transformations):
        W = np.random.normal(0, 1, size=list_dimensions[i])
        list_W.append(W)

    # TODO: add more non-linear functions if required
    if cfg_file["transformation"]["non_linear"] == "sigmoid":
        F_non_linear = sigmoid

    return list_W, F_non_linear

def generate_synthetic_lines(cfg_path, verbose = True):
    """
        Generates elongated clusters.
        Arguments:
            cfg_file: config file with the specifications to apply rotations and translations to the clusters.
        Outputs:
            X_train: train data set with the elongated clusters
            Y_train: labels for the training set.
            X_test: test set with the elongated clusters.
            Y_test: labels for the test set.

    """


    # laod config file
    cfg_file = load_config(cfg_path)

    # create a path for the log directory
    path_log_dir = cfg_file["data"]["plots"]["path"] + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    if not os.path.isdir(path_log_dir):
        os.mkdir(path_log_dir)

    log_tensorboard = path_log_dir + "/tensorboard/"

    writer = SummaryWriter(log_dir=log_tensorboard)

    n_samples_train = cfg_file["data"]["train"]["num_samples"]
    rotation = cfg_file["data"]["train"]["rotation"]

    if verbose:
        print("Generating lines...")

    separation = 5 # separation among each elongated cluster
    # list for the translation of each cluster
    d = separation * np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])
    # list with the rotation angle for each cluster
    #theta_list = [1.434, 4.302, 5.404, 5.657, 0.463, 5.293, 0.64, 0.238, 3.073] # rotation 1
    theta_list = [2.193, 5.189, 4.584, 3.426, 5.819, 0.229, 4.085, 1.597, 1.202] # rotation 2
    # list with distribution for the clusters
    # p_dist = [0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.2, 0.2, 0.4] # heavy imbalance
    # p_dist = [0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2]  # soft imbalance
    p_dist = 9 *[1/9] # uniform distribution
    X_train = None
    Y_train = None
    for i in range(9):
        # rng = np.random.RandomState(2) # regular shapes
        rng = np.random.RandomState(i+8) # different shapes
        n_samples_i = int(p_dist[i] * n_samples_train )
        X_pre = np.dot(rng.rand(2, 2), rng.randn(2, n_samples_i)).T
        if rotation: # with rotation
            # define rotation matrix
            r = np.zeros((2, 2))
            theta = theta_list[i]
            r[0, 0] = np.cos(theta)
            r[0, 1] = np.sin(theta)
            r[1, 0] = -1 * np.sin(theta)
            r[1, 1] = np.cos(theta)
            X = np.dot(X_pre, r) + d[i, :]
        else: # without rotation
            X = X_pre + d[i,:]

        # compute the centers by means of PCA
        pca = PCA(n_components=2)
        pca.fit(X)
        centers = pca.mean_

        # compute the labels
        Y = i * np.ones(X.shape[0])
        if i == 0:
            X_train = X
            Y_train = Y
            centers_train = centers
        else:
            X_train = np.vstack((X_train, X))
            Y_train = np.hstack((Y_train, Y))
            centers_train = np.vstack((centers_train, centers))

    if cfg_file["data"]["transformation"]:
        if verbose:
            print("Creating non-linear transformation...")

        # transformations to bring lower-dimensional data to a higher-dimensional space
        list_W, F_non_linear = generate_list_W_and_F(cfg_file)

        if verbose:
            print("Applying transformation to the data...")
        X_train = transform_low_to_high(X_train, F_non_linear, list_W)
        centers_train = transform_low_to_high(centers_train, F_non_linear, list_W)

    if cfg_file["data"]["normalize"]:
        if verbose:
            print("Normalizing data...")
        X_train, centers_train = normalize_data(X_train, centers_train)


    X_test = X_train
    Y_test = Y_train
    centers_test = centers_train

    if cfg_file["data"]["save"]:
        if verbose:
            print("Saving data...")
        # save the training data
        np.save(cfg_file["data"]["train"]["path"] + "X_train", X_train)
        np.save(cfg_file["data"]["train"]["path"] + "Y_train", Y_train)
        np.save(cfg_file["data"]["train"]["path"] + "centers_train", centers_train)
        # save the test data
        np.save(cfg_file["data"]["test"]["path"] + "X_test", X_test)
        np.save(cfg_file["data"]["test"]["path"] + "Y_test", Y_test)
        np.save(cfg_file["data"]["test"]["path"] + "centers_test", centers_test)


    if verbose:
        print("Data generation completed!")

    return X_train, Y_train, X_test, Y_test


def generate_synthetic_clusters(cfg_path, verbose = True):
    """
        Generates gaussian cluster blobs.
        Arguments:
            cfg_file: config file with the specifications to generate the gaussian blobs.
        Outputs:
            X_train: train data set with the gaussian cluster blobs.
            Y_train: labels for the training set.
            X_test: test data set with the gaussian cluster blobs.
            Y_test: labels for the test set.

    """

    # laod config file
    cfg_file = load_config(cfg_path)

    # create a path for the log directory
    path_log_dir = cfg_file["data"]["plots"]["path"] + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    if not os.path.isdir(path_log_dir):
        os.mkdir(path_log_dir)

    log_tensorboard = path_log_dir + "/tensorboard/"

    writer = SummaryWriter(log_dir=log_tensorboard)

    num_centers = cfg_file["clusters"]["num_centers"]
    center_box = tuple(cfg_file["clusters"]["center_box"])
    cluster_std = cfg_file["clusters"]["cluster_std"]
    random_state = cfg_file["clusters"]["random_state"]
    n_samples_train = cfg_file["data"]["train"]["num_samples"]
    n_samples_test = cfg_file["data"]["test"]["num_samples"]
    dim = cfg_file["clusters"]["dim"]

    if verbose:
        print("Generating clusters...")

    # generate training data given the hyperparameters with the help of make_blobs function
    X_train_raw, Y_train, centers_train = make_blobs(n_samples = n_samples_train, n_features = dim,
                                                     centers = num_centers, center_box = center_box,
                                                     cluster_std = cluster_std, shuffle = True,
                                                     random_state = random_state, return_centers = True)

    # generate the test data with the same centers as the training data
    X_test_raw, Y_test, centers_test = make_blobs(n_samples = n_samples_test, n_features = dim,
                                                  centers = centers_train, cluster_std = cluster_std,
                                                  shuffle = True, random_state = random_state, return_centers = True)

    # code to handle when the clusters are too close
    while np.linalg.norm(centers_train[0] - centers_train[1]) <= 2*cluster_std:
        random_state += 1
        # generate training data given the hyperparameters with the help of make_blobs function
        X_train_raw, Y_train, centers_train = make_blobs(n_samples=n_samples_train, n_features=dim,
                                                         centers=num_centers, center_box=center_box,
                                                         cluster_std=cluster_std, shuffle=True,
                                                         random_state=random_state, return_centers=True)

        # generate the test data with the same centers as the training data
        X_test_raw, Y_test, centers_test = make_blobs(n_samples=n_samples_test, n_features=dim,
                                                      centers=centers_train, cluster_std=cluster_std,
                                                      shuffle=True, random_state=random_state, return_centers=True)

    if cfg_file["data"]["normalize"]:
        if verbose:
            print("Normalizing data...")
        X_train, centers_train = normalize_data(X_train_raw, centers_train)
        X_test, centers_test = normalize_data(X_test_raw, centers_test)
    else:
        X_train = X_train_raw
        X_test = X_test_raw


    if cfg_file["data"]["save"]:
        if verbose:
            print("Saving data...")
        # save the training data
        np.save(cfg_file["data"]["train"]["path"] + "X_train_raw", X_train_raw)
        np.save(cfg_file["data"]["train"]["path"] + "X_train", X_train)
        np.save(cfg_file["data"]["train"]["path"] + "Y_train", Y_train)
        np.save(cfg_file["data"]["train"]["path"] + "centers_train", centers_train)
        # save the test data
        np.save(cfg_file["data"]["test"]["path"] + "X_test_raw", X_test_raw)
        np.save(cfg_file["data"]["test"]["path"] + "X_test", X_test)
        np.save(cfg_file["data"]["test"]["path"] + "Y_test", Y_test)
        np.save(cfg_file["data"]["test"]["path"] + "centers_test", centers_test)

    if cfg_file["data"]["plot"]:
        if verbose:
            print("Plotting data...")

        # visualize training data in 2D

        title = "Training data (raw)"
        writer.add_figure('01 Visualization 2D training data (raw) ',
                          plot_X2D_visualization(X_train_raw, Y_train,
                                                 title = title, num_classes = num_centers))

        title = "Training data"
        writer.add_figure('01.1 Visualization 2D training data',
                          plot_X2D_visualization(X_train, Y_train, title=title, num_classes = num_centers))

        # visualize test data in 2D

        title = "Test data (raw)"
        writer.add_figure('02 Visualization 2D test data (raw)',
                          plot_X2D_visualization(X_test_raw, Y_test,
                                                 title=title, num_classes = num_centers))
        title = "Test data"
        writer.add_figure('02.1 Visualization 2D test data',
                          plot_X2D_visualization(X_test, Y_test, title=title, num_classes = num_centers))

        writer.add_figure('02.1 Visualization 2D test data',
                          plot_X2D_visualization(X_test, Y_test, title=title, num_classes = num_centers))


    if verbose:
        print("Data generation completed!")

    return X_train, Y_train, X_test, Y_test

def generate_synthetic_functions(cfg_path, verbose = True):
    """
       Generates clusters from sine wave functions (in 2D) and apply linear and non-linear transformations.
       Arguments:
           cfg_file: config file with the specifications to define the functions and apply the transformations.
       Outputs:
           X_train: train data set with the clusters generated from sine wave functions.
           Y_train: labels for the training set.
           X_test: test data set with the clusters generated from sine wave functions.
           Y_test: labels for the test set.

   """

    # laod config file
    cfg_file = load_config(cfg_path)

    interval_F1 = tuple(cfg_file["F1"]["interval"])
    interval_F2 = tuple(cfg_file["F2"]["interval"])


    # create a path for the log directory
    path_log_dir = cfg_file["data"]["plots"]["path"] + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    if not os.path.isdir(path_log_dir):
        os.mkdir(path_log_dir)

    log_tensorboard = path_log_dir + "/tensorboard/"

    writer = SummaryWriter(log_dir = log_tensorboard)

    if verbose:
        print ("Defining functions with parameters from config file: {}".format(cfg_file["name"]))

    # functions for training data
    F1_train = FunctionSin(cfg_file["F1"]["amp"], cfg_file["F1"]["frec"], interval_F1, cfg_file["F1"]["shift"],
                        cfg_file["F1"]["char_to_plot"], cfg_file["F1"]["color_to_plot"])

    F2_train = FunctionSin(cfg_file["F2"]["amp"], cfg_file["F2"]["frec"], interval_F2, cfg_file["F2"]["shift"],
                           cfg_file["F2"]["char_to_plot"], cfg_file["F2"]["color_to_plot"])

    # functions for test data
    F1_test = FunctionSin(cfg_file["F1"]["amp"], cfg_file["F1"]["frec"], interval_F1, cfg_file["F1"]["shift"],
                          cfg_file["F1"]["char_to_plot"], cfg_file["F1"]["color_to_plot"])

    F2_test = FunctionSin(cfg_file["F2"]["amp"], cfg_file["F2"]["frec"], interval_F2, cfg_file["F2"]["shift"],
                          cfg_file["F2"]["char_to_plot"], cfg_file["F2"]["color_to_plot"])

    # compute the centers for the functions
    cx_F1 = (interval_F1[0]+ interval_F1[1])/2
    cy_F1 = cfg_file["F1"]["shift"]
    cx_F2 = (interval_F2[0] + interval_F2[1]) / 2
    cy_F2 = cfg_file["F2"]["shift"]
    centers_train = np.array( [[cx_F1, cy_F1], [cx_F2, cy_F2]])
    centers_test = np.array( [[cx_F1, cy_F1], [cx_F2, cy_F2]])


    if verbose:
        print("Generating data...")

    # generation of the training and test data
    F1_train.generate_data(cfg_file["F1"]["train_num_samples"])
    F2_train.generate_data(cfg_file["F2"]["train_num_samples"])
    F1_test.generate_data(cfg_file["F1"]["test_num_samples"])
    F2_test.generate_data(cfg_file["F2"]["test_num_samples"])

    # TODO: adapt code for list of functions if required
    # define lists with the defined functions
    list_F_train = [F1_train, F2_train]
    list_F_test = [F1_test, F2_test]

    num_classes = len(list_F_train)

    # get raw data from the functions
    X_train_raw = np.vstack((F1_train.vec, F2_train.vec))
    X_test_raw = np.vstack((F1_test.vec, F2_test.vec))

    if cfg_file["data"]["transformation"]:
        if verbose:
            print("Creating non-linear transformation...")

        # transformations to bring lower-dimensional data to a higher-dimensional space
        list_W, F_non_linear = generate_list_W_and_F(cfg_file)

        if verbose:
            print("Applying transformation to the data...")
        X_train = transform_low_to_high(X_train_raw, F_non_linear, list_W)
        X_test = transform_low_to_high(X_test_raw, F_non_linear, list_W)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    if cfg_file["data"]["normalize"]:
        if verbose:
            print("Normalizing data...")
        X_train, centers_train = normalize_data(X_train, centers_train, verbose = True)
        X_test, centers_test = normalize_data(X_test, centers_test, verbose = True)

    # labels for training data
    labels_train0 = np.zeros(F1_train.vec.shape[0])
    labels_train1 = np.ones(F2_train.vec.shape[0])
    Y_train = np.hstack((labels_train0, labels_train1))
    # labels for test data
    labels_test0 = np.zeros(F1_test.vec.shape[0])
    labels_test1 = np.ones(F2_test.vec.shape[0])
    Y_test = np.hstack((labels_test0, labels_test1))

    # save the training and test data
    if cfg_file["data"]["save"]:
        if verbose:
            print("Saving data...")
        np.save(cfg_file["data"]["train"]["path"] + "X_train_raw", X_train_raw)
        np.save(cfg_file["data"]["train"]["path"] + "X_train", X_train)
        np.save(cfg_file["data"]["train"]["path"] + "Y_train", Y_train)
        np.save(cfg_file["data"]["train"]["path"] + "centers_train", centers_train)
        np.save(cfg_file["data"]["test"]["path"] + "X_test_raw", X_test_raw)
        np.save(cfg_file["data"]["test"]["path"] + "X_test", X_test)
        np.save(cfg_file["data"]["test"]["path"] + "Y_test", Y_test)
        np.save(cfg_file["data"]["test"]["path"] + "centers_test", centers_test)


    # run k-means on the training data
    # centers_train_raw, labels_train_raw = k_means(X_train_raw, n_clusters = len(list_F_train))
    centers_train_kmeans, labels_kmeans_train = k_means(X_train, centers_init = None, n_clusters = len(list_F_train))

    # run k-means on the test data
    # centers_test_raw, labels_test_raw = k_means(X_test_raw, n_clusters = len(list_F_test))
    centers_test_kmeans, labels_kmeans_test = k_means(X_test, centers_init = None, n_clusters=len(list_F_test))

    if cfg_file["data"]["plot"]:
        if verbose:
            print("Plotting data...")

        # plot the 2D visualization of the training data generated
        titles = ["Functions and noisy data", "Training data", "K-means on training data"]
        writer.add_figure('01 Visualization of training data generated ',
                          plot_2D_visualization_generation_functions(list_functions = [F1_train, F2_train],
                                                                     X = X_train, labels = Y_train,
                                                                     num_classes = num_classes,
                                                                     labels_kmeans = labels_kmeans_train,
                                                                     cluster_centers = centers_train_kmeans,
                                                                     titles = titles))

        # plot the 2D visualization of the test data generated
        titles = ["Functions and noisy data", "Test data", "K-means on test data"]
        writer.add_figure('02 Visualization of test data generated ',
                          plot_2D_visualization_generation_functions(list_functions = [F1_test, F2_test],
                                                                     X = X_test, labels = Y_test,
                                                                     num_classes = num_classes,
                                                                     labels_kmeans = labels_kmeans_test,
                                                                     cluster_centers = centers_test_kmeans,
                                                                     titles = titles))
    if verbose:
        print("Data generation completed!")

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")
    args = ap.parse_args()

    print("DATA_SET: ", DATA_SET)
    if DATA_SET == "synthetic_functions": # sine wave clusters
        generate_synthetic_functions(cfg_path = args.config_path)
    elif DATA_SET == "synthetic_clusters": # isotropic gaussian blob clusters
        generate_synthetic_clusters(cfg_path = args.config_path)
    elif DATA_SET == "synthetic_lines": # elongated clusters with rotations and translations
        generate_synthetic_lines(cfg_path = args.config_path)