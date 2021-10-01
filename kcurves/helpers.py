# libraries
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import csv
from torch import linalg as LA
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils.linear_assignment_ import linear_assignment
from constants import DEVICE
import pylab as pl
from matplotlib import collections  as mc
from matplotlib import colors as mcolors

def imshow(list_images):
    """
    Plots the original image and the reconstructed image side by side.
    Arguments:
        list_images: list of pairs of images, the pair is composed of the original and reconstructed image.
    Outputs:
        fig: the figure created after plotting the images from the list.
    """
    num_images = len(list_images)
    fig = plt.figure(figsize=(10, 10))


    for idx_img, pair_image in enumerate(list_images):
        img_original, img_reconstructed = pair_image

        # plot the original image
        ax1 = fig.add_subplot(num_images, 2, 2*idx_img+1)
        plt.imshow(img_original, cmap='gray')
        if idx_img == 0:
            ax1.set_title("Original image")
        ax1.set_xticks([])
        ax1.set_yticks([])
        # plot the reconstructed image
        ax2 = fig.add_subplot(num_images, 2, 2*idx_img + 2)
        plt.imshow(img_reconstructed, cmap='gray')
        if idx_img == 0:
            ax2.set_title("Reconstructed image")
        ax2.set_xticks([])
        ax2.set_yticks([])

    return fig

def plot_X2D_visualization(X, labels, title, num_classes, cluster_centers=None):
    """
    Plots the visualization of the data in 2D. If the dimension of the data is higher
    than 2, then it applies PCA to the data.
    Arguments:
        X: data to be plotted.
        labels: true labels of the data.
        title: string with the title of the graphic.
        num_classes: Number of classes.
        cluster_centers: Cluster centers, if any, to be plotted.
    Outputs:
        fig: the figure created after plotting of the data in 2D.

    """
    if X.shape[1] > 2:
        X_2D = run_PCA(X)
    else:
        X_2D = X

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # loop to plot the clusters depending on the given labels
    for i in range(num_classes):
        idx = np.where(labels == i)
        ax.scatter(X_2D[idx, 0], X_2D[idx, 1], cmap='viridis')

    # plot the center of the clusters if any
    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, alpha=0.5)

    list_legend = ["class {}".format(i) for i in range(num_classes)]
    ax.legend(list_legend)
    ax.set_title(title)

    return fig

# TODO: improve readability of this function
def plot_2D_visualization_clusters(list_X, labels, list_predictions, list_rep, list_inter,
                                   list_titles, list_metrics, num_classes,type_rep, x_label,
                                   visualization):
    """
    Visualizes outputs and inputs of the several algorithms.
    Arguments:
        list_X: List [X_input, X_reconstructed, H, softmax(H)], where:
            -X_input is the input data
            -X_reconstructed is the reconstructed data(output of the auto-encoder)
            -H is the latent vector.
            -softmax(H) is the softmax over the latent vector.
        labels: true labels of the data.
        list_predictions: List []
        list_rep:
        list_inter:
        list_titles:
        list_metrics:
        num_classes: number of classes of the data.
        type_rep:
        x_label:
        visualization:
    Outputs:
        fig: figure with all the plots.
    """
    cdict_points = {0: 'silver', 1: 'lightcoral', 2: 'peachpuff', 3: 'bisque', 4: 'cornsilk',
                    5: 'lightgreen', 6: 'lightcyan', 7: 'lightskyblue', 8: 'thistle', 9: 'magenta',
                    10: 'ivory', 11: 'azure', 12: 'papayawhip', 13: 'lightcoral', 14:'lightsalmon',
                    15:'khaki', 16:'orchid', 17:'olive', 18:'blue', 19:'red'}
    cdict_rep = {0: 'black', 1: 'red', 2: 'saddlebrown', 3: 'darkorange', 4: 'gold',
                 5: 'green', 6: 'cyan', 7: 'deepskyblue', 8: 'darkviolet', 9: 'darkmagenta',
                 10: 'wheat', 11: 'lightcyan', 12: 'moccasin', 13: 'coral', 14: 'darksalmon',
                 15:'darkkhaki', 16:'mediumorchid', 17:'darkolivegreen', 18:'mediumblue', 19:'maroon'}

    list_X_full = []
    list_dim = []
    size_batch = list_X[0].shape[0]
    for i in range(len(list_X)):
        X = list_X[i]
        dim = X.shape[1]
        list_dim.append(dim)
        rep = list_rep[i]
        s_inter = list_inter[i]
        if type_rep == "points":
            X_full = np.vstack((X, rep))
        elif type_rep == "segments":
            X_full = np.vstack((X, rep[:,:dim]))
            if i == 1 or i ==2:
                X_full = np.vstack((X_full, rep[:, dim:]))
            if s_inter is not None:
                X_full = np.vstack((X_full, s_inter))
        print("i = {}, X_full shape = {}".format(i, X_full.shape))
        list_X_full.append(X_full)

    for i in range(len(list_X_full)):
        if list_X_full[i].shape[1] > 2:
            if visualization == "PCA":
                list_titles[i] = list_titles[i] + "(PCA)"
                print("Running PCA...")
                list_X_full[i] = run_PCA(list_X_full[i])
            else:
                list_titles[i] = list_titles[i] + "(T-SNE)"
                print("Runnning T-SNE...")
                list_X_full[i] = run_TSNE(list_X_full[i])

    for i in range(len(list_X_full)):
        X_full = list_X_full[i]
        print("i = {}, X_full shape = {}".format(i, X_full.shape))
        dim = list_dim[i]
        X = X_full[:size_batch, :]
        if type_rep == "points" or i == 0 or i ==3:
            rep = X_full[size_batch:size_batch+num_classes,:]
            s_inter = None
        elif type_rep == "segments":
            rep = X_full[size_batch:size_batch+2*num_classes, :]
            s_inter = X_full[size_batch+2*num_classes:, :]

        list_X[i] = X
        list_rep[i] = rep
        list_inter[i] = s_inter


    list_cx = [0, 0, 0, 1, 1, 1]
    list_cy = [0, 1, 2, 0, 1, 2]

    per = [3,2,1,4,5,0]

    plt.switch_backend('agg')
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for i in range(len(list_X)):
        cx = list_cx[per[i]]
        cy = list_cy[per[i]]
        X = list_X[i]
        rep = list_rep[i]
        s_inter = list_inter[i]
        title = list_titles[i]

        # loop to plot the clusters depending on the given labels
        for j in range(num_classes):
            if i == 0 or i==5:
                idx = np.where(labels == j)
            elif i == 3:
                idx = np.where(list_predictions[2] == j)
            elif i == 4:
                idx = np.where(list_predictions[3] == j)
            else:
                idx = np.where(list_predictions[4] == j)
            if cx < 1:
                #axs[cx, cy].scatter(X[idx, 0], X[idx, 1], color = cdict_points[j], linewidths = 0.1, alpha = 0.1)
                #axs[cx, cy].set_title(title)
                axs[cy].scatter(X[idx, 0], X[idx, 1], color = cdict_points[j], linewidths = 0.1, alpha = 0.1)
                axs[cy].set_title(title)


        # first point of the segments
        s1 = rep[:num_classes, :]

        if i!=0 and i!=5:
            for j in range(num_classes):
                if cx < 1:
                    #axs[cx, cy].scatter(s1[j, 0], s1[j, 1], marker= '^', color=cdict_rep[j], linewidths=2, alpha=1)
                    axs[cy].scatter(s1[j, 0], s1[j, 1], marker= '^', color=cdict_rep[j], linewidths=2, alpha=1)

        #if i == 3:
        #    axs[cx, cy].legend(list_metrics[2],  bbox_to_anchor=(1.3, 1.3 ))

        if i == 1 or i == 2 or i==4:
            if type_rep == "segments":
                # second point of the segments and interpolation points
                s2 = rep[num_classes:, :]

                # code to draw points from interpolation
                for j in range(num_classes):
                    if cx <1:
                        #axs[cx, cy].scatter(s2[j, 0], s2[j, 1], marker= 'v', color=cdict_rep[j], linewidths=2, alpha=1)
                        axs[cy].scatter(s2[j, 0], s2[j, 1], marker= 'v', color=cdict_rep[j], linewidths=2, alpha=1)

                    idx_j = np.arange(j, s_inter.shape[0], num_classes)
                    if cx<1:
                        #axs[cx, cy].scatter(s_inter[idx_j, 0], s_inter[idx_j, 1], marker= '*', color=cdict_rep[j], linewidths=1, alpha=1)
                        axs[cy].scatter(s_inter[idx_j, 0], s_inter[idx_j, 1], marker= '*', color=cdict_rep[j], linewidths=1, alpha=1)


            #if i==4:
            #    axs[cx, cy].legend(list_metrics[3],  bbox_to_anchor=(1.3, 1.3 ))
            #else:
            #    axs[cx, cy].legend(list_metrics[4], bbox_to_anchor=(1.3, 1.3))
        if cx<1:
            #axs[cx, cy].set_aspect('equal')
            #axs[cx, cy].set_xlabel(x_label)

            axs[cy].set_aspect('equal')
            axs[cy].set_xlabel(x_label)


    # -----------------------------Frozen code-------------------------------
    # x_i, y_i, z_i_ent, z_i_dist = list_vars
    # # plot graphic for entropy
    # im_20 = axs[2, 0].contourf(x_i ,y_i, z_i_ent, levels = levels_contour)
    # axs[2, 0].set_title(titles[4])
    # fig.colorbar(im_20, ax = axs[2, 0])
    # # plot  graphic for distance
    # im_21 = axs[2, 1].contourf(x_i, y_i, z_i_dist, levels=levels_contour)
    # axs[2, 1].set_title(titles[5])
    # fig.colorbar(im_21, ax=axs[2, 1])

    return fig

def plot_2D_visualization_generation_functions(list_functions, X, labels, num_classes, labels_kmeans,
                                               cluster_centers, titles):
    """
    Visualizes generation of synthetic data via functions (functions and data generated)
    and the result of applying k-means on the synthetic data.
    :param list_functions:
    :param X:
    :param labels: true labels.
    :param num_classes: number of classes (clusters).
    :param labels_kmeans: labels after applying vanilla k-means.
    :param cluster_centers: cluster centers.
    :param titles: titles for each plot.
    :return:
        -fig: Figure with the plots.
    """


    if X.shape[1] > 2:
        X = run_PCA(X)

    plt.switch_backend('agg')
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # make plot for functions
    list_legends = []
    for idx, f in enumerate(list_functions):
        axs[0, 0].plot(f.x, f.y_noisy, f.char_to_plot) # Plot noisy data
        list_legends.append("Noisy data from Function F{}".format(idx+1))
        axs[0, 0].plot(f.x, f.y, color=f.color_to_plot) # Plot f(x)
        list_legends.append("Function F{}".format(idx+1))

    axs[0,0].set_title(titles[0])
    axs[0,0].legend(list_legends)

    # make plot for synthetic data (training or test data)
    for i in range(num_classes):
        idx = np.where(labels == i)
        axs[0, 1].scatter(X[idx, 0], X[idx, 1], cmap = 'viridis')
        axs[0, 1].set_title(titles[1])

    # make plot for k-means on synthetic data (training or test data)
    for i in range(num_classes):
        idx = np.where(labels_kmeans == i) # plot the clusters depending on the given labels
        axs[1, 0].scatter(X[idx, 0], X[idx, 1], cmap = 'viridis')

    # plot the center of the clusters if any
    if cluster_centers is not None:
        axs[1, 0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, alpha=0.5)

    list_legend = ["class {}".format(i) for i in range(num_classes)]
    axs[1, 0].legend(list_legend)
    axs[1, 0].set_title(titles[2])

    return fig




def plot_functions(list_functions, title):
    """
    Plots noisy data and function given a list of functions.
    Arguments:
        list_functions: list with all the functions to be plotted.
        title: title of the graphic.
    Outputs:
        fig: Figure with the plot with the functions plotted.

    """
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    list_legends = []
    for idx, f in enumerate(list_functions):
        ax.plot(f.x, f.y_noisy, f.char_to_plot) # Plot noisy data
        list_legends.append("Noisy data from Function F{}".format(idx+1))
        ax.plot(f.x, f.y, color=f.color_to_plot) # Plot f(x)
        list_legends.append("Function F{}".format(idx+1))

    ax.set_title(title)
    ax.legend(list_legends)

    return fig
def get_interpolation(s1, s2, num_points):
    """
     Computes equidistant (finite) points between two ending points.
     Arguments:
         s1: initial point of the segment.
         s2: end point of the segment.
         num_points: number of points within the segment to be computed.
     Outputs:
         s_inter: tensor with the equidistant points.

     """
    dim0 = s1.shape[0]
    dim1 = s1.shape[1]
    s1 = s1.repeat(num_points - 1, 1)
    s2 = s2.repeat(num_points - 1, 1)

    s_alphas = torch.tensor(np.linspace(1/num_points,1 ,num_points)[:-1]).type(torch.FloatTensor).to(DEVICE)
    s_alphas = s_alphas.view(1,-1).repeat(dim0, 1).T.reshape(-1,1).repeat(1, dim1)
    print("s1 shape: ", s1.shape)
    print("s2 shape: ", s2.shape)
    print("s_alphas shape: ", s_alphas.shape)

    s_inter = s_alphas * s1 + (1 - s_alphas) * s2

    return s_inter

def transform_low_to_high(X_low, F_non_linear, list_W):
    """
    Maps the values from a low dimensional space to a higher dimensional space,
    by means of the following formula (applied multiple times):
        X_high = F_non_linear (U * F_non_linear (W * X_low))
    where X_high are the resulting vectors in the high dimensional space and
    X_low are the input vectors from the low dimensional space.
    See below for the other parameters.
    Arguments:
        X_low: data in low dimensional space.
        list_W: list of matrices for the linear transformation to be applied.
        F_non_linear: non linear function applied in the formula.
    Outputs:
        X_high: data in a higher dimensional space.
    """
    # apply the transformation to the normalized data by using the matrices stored in the list
    vec_temp = X_low.T
    for W in list_W:
        vec_temp = F_non_linear(W @ vec_temp)

    X_high = vec_temp.T

    return X_high

def run_PCA(X, n_components=2):
    """
    Runs PCA given a matrix data (numpy array).
    Arguments:
        X: data in the in high dimension.
        n_components: number of components to be considered in the PCA.
    Outputs:
        X_pca: projection of the data in n_components as a result of PCA.
    """
    pca = PCA(n_components=n_components)
    pca_trans = pca.fit(X)
    X_pca = pca_trans.transform(X)

    return X_pca

def run_TSNE(X, n_components=2):
    """
    Runs the TSNE (https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) algorithm to
    reduce the dimensionality of a given data matrix.
    Arguments:
        X: data in the in high dimension.
        n_components: number of components to be considered for the TSNE.
    Outputs:
        X_tsne: data matrix after applying TSNE.
    """
    X_tsne = TSNE(n_components = n_components, random_state=0).fit_transform(X)

    return X_tsne

def sigmoid(x):
    """
    Applies the sigmoid function element-wise to a given numpy array.
    Arguments:
        x: numpy array.
    Output:
        _ : numpy array with the sigmoid function values computed.

    """
    return 1 / (1 + np.exp(-x))

def get_softmax(X_2D):
    """
    Computes the softmax of a numpy array or a torch tensor.
    Arguments:
        X_2D: 2-dimensional array.
    Outputs:
        X_2D_softmax: numpy array or torch tensor after applying the softmax to X_2D.
    """

    if type(X_2D) is np.ndarray:
        T_2D = torch.from_numpy(X_2D).to(DEVICE)
    else:
        T_2D = X_2D

    T_2D_softmax = F.softmax(T_2D, dim = 1)
    X_2D_softmax = T_2D_softmax.cpu().detach().numpy()

    return X_2D_softmax


def softmax_pair(x, y, absolute =True, alpha=None):
    """
    Computes the softmax of a pair (x,y).
    Arguments:
        x: x-coordinate (float).
        y: y-coordinate (float).
        absolute: boolean flag to indicate if the absolute value operation must be applied.
        alpha: temperature parameter (float).
    Outputs:
        a: softmax in the  x-coordinate.
        b: softmax in the y-coordinate.
    """
    if absolute:
        x = np.absolute(x)
        y = np.absolute(y)
    if alpha is not None:
        a = np.exp(-alpha * x) / (np.exp(-alpha * x) + np.exp(-alpha * y))
        b = np.exp(-alpha * y) / (np.exp(-alpha * x) + np.exp(-alpha * y))
    else:
        a = np.exp(x) / (np.exp(x) + np.exp(y))
        b = np.exp(y) / (np.exp(x) + np.exp(y))

    return a, b


def entropy_pair(x, y):
    """
    Computes the cross entropy of a pair(x,y).
    Arguments:
        x: x-coordinate (float).
        y: y-coordinate (float).
    Outputs:
        _: entropy of the pair.
    """
    return -x * np.log(x) - y * np.log(y)

def distance_axes_pair(x, y):
    """
    Computes the distance d as follows:
        d = exp^(x)/(exp^(x) + exp^(y)) * |x| + exp^(y)/(exp^(x) + exp^(y)) * |y|.
    Arguments:
        x: x-coordinate (float).
        y: y-coordinate (float).
    Outputs:
        _: distance d as described above given the pair (x,y).
    """
    a, b = softmax_pair(x, y)
    return a * np.absolute(x) + b * np.absolute(y)

# TODO: Change the name to pairwise_distance_points
def pairwise_distances(x, y):
    """
    Compute all the distances between the points x[i,:] and y[j,:], where x is a
    matrix with N points and y a matrix with M points, with 0<=i<N, 0<=j<N.
    Arguments:
        x: tensor of dimensions (N,d).
        y: tensor of dimensions (M,d).
    Output:
        dist: a (N,M) tensor where dist[i,j] is the square L2-norm.
              between x[i,:] and y[j,:] => dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)

def pairwise_distances_segments(X, s):
    """
    Compute all the distances between the points x[i,:] and the segments
    s[j,:], where X is a matrix with N points and s is a matrix with M segments,
    with 0<=i<N, 0<=j<N.
    Arguments:
        X: tensor of dimensions (N,d).
        s: tensor of dimensions (M,2d).
    Output:
        dist: a (N,M) tensor where dist[i,j] is the distance between the point x[i,:] and the segment s[j,:].
    """

    dim = X.shape[1]

    # s2 - s1
    diff = s[:, dim:] - s[:, :dim]
    norm = LA.norm(diff, dim=1) ** 2
    norm = norm.view(-1, 1).repeat(1, dim)
    eps = 1e-9
    diff_norm = diff / (norm + eps)

    for i in range(diff.shape[0]):
        s1_i = s[i, :dim].repeat(X.shape[0], 1)
        diff_norm_i = diff_norm[i, :].repeat(X.shape[0], 1)
        diff_i = diff[i, :].repeat(X.shape[0], 1)
        mult = (X - s1_i) * diff_norm_i
        dot = torch.sum(mult, dim = 1)
        t = torch.clamp(dot, min = 0.0, max = 1.0)
        t_i = t.reshape(-1, 1).repeat(1, dim)
        t_opt = s1_i + t_i * (diff_i)
        dist_i = torch.sum((X - t_opt) ** 2, dim=1).view(-1, 1)
        if i == 0:
            dist = dist_i
        else:
            dist = torch.cat((dist, dist_i), 1)

    return torch.clamp(dist, 0.0, np.inf)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    Arguments:
        path: path to YAML configuration file.
    Outputs:
        cfg: configuration dictionary.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg

def create_writer(path_log_dir):
    """
    Creates a writer for tensorboard given a path.
    Arguments:
        path_log_dir: path for the writer.
    Outputs:
        writer: writer object created.
    """

    if not os.path.isdir(path_log_dir):
        os.mkdir(path_log_dir)

    log_tensorboard = path_log_dir + "/tensorboard/"
    writer = SummaryWriter(log_dir=log_tensorboard)

    return writer

def get_hyperparameters(cfg_path):
    """
    Extracts the hyperparameters from the config file to make a dictionary out of them.
    Arguments:
        cfg_path: path of the config file.
    Outputs:
        dic_hyperparameters: dictionary with the hyperparameters.
    """

    cfg_file = load_config(cfg_path)

    lr = cfg_file["train"]["lr"]
    num_epochs = cfg_file["train"]["num_epochs"]
    type_rep = cfg_file["train"]["type_rep"]
    type_loss = cfg_file["train"]["type_loss"]
    alpha_init = cfg_file["train"]["alpha_init"]
    beta_type = cfg_file["train"]["beta_type"]
    beta_init = cfg_file["train"]["beta_init"]
    gamma_type = cfg_file["train"]["gamma_type"]
    gamma_init = cfg_file["train"]["gamma_init"]
    init_strategy = cfg_file["train"]["init_strategy"]
    dic_hyperparameters = {"type_rep": type_rep, "type_loss": type_loss,
                           "init_strategy": init_strategy,
                           "beta_type": beta_type, "beta_init": beta_init,
                           "gamma_type": gamma_type, "gamma_init": gamma_init,
                           "num_epochs": num_epochs,
                           "alpha_init": alpha_init, "lr": lr}

    return dic_hyperparameters

def make_string_from_dic(dic):
    """
    Makes a string out of a dictionary, by relating the keys and values with "=" symbol and by
    using a "_" as separator.
        For example:
        dic = {key1: val1, key2: val2, key3: val3 } => str_ = "key1=val1_key2=val2_key3=val3"
    Arguments:
        dic: dictionary where the transformation will be applied.
    Outputs:
        str_: string that represents the given dictionary.
    """
    str_ = ""
    for key, value in dic.items():
        str_ += key + "=" +str(value)+"_"

    str_ = str_[:-1] # remove the last '_' from the string

    return str_

def add_zeros (n, lim):
    """
    Adds zeros at the beginning of a number w.r.t to lim.
    Arguments:
        n: number to add zeros at the beginning.
        lim: upper bound for n.
    Outputs:
        num: string with the number with zeros at the beginning.
    """
    num = ""
    # put 0's at the beginning w.r.t. lim
    dif = int(np.log10(lim)) - int(np.log10(n))
    for j in range(dif):
        num += "0"
    # add the number given by n
    num += str(n)

    return num

def freeze_module (module):
    """
    Freezes a given layer.
    Arguments:
        module: name of the layer to be frozen.
    """
    for param in module.parameters():
        param.requires_grad = False

    return

def unfreeze_module(module):
    """
    Unfreezes a given layer.
    Arguments:
        module: name of the layer to be unfrozen.
    """
    for param in module.parameters():
        param.requires_grad = True

    return

def Read_Two_Column_File(file_name, type_np):
    """
    Reads a file with two column format.
    Arguments:
        file_name: name of the file to be read.
        type_np: data type of the columns.
    Outputs:
        X.T: numpy matrix with the data (from the file) stored.
    """

    with open(file_name, 'r') as f_input:
        csv_input = csv.reader(f_input, delimiter=' ', skipinitialspace=True)
        x = []
        y = []
        for cols in csv_input:
            if type_np == 'float':
                x.append(float(cols[0]))
                y.append(float(cols[1]))
            else:
                x.append(int(cols[0]))
                y.append(int(cols[1]))

        X = np.vstack((x, y))
    return X.T

def Read_One_Column_File(file_name, type_np):
    """
    Reads a file with one column format.
    Arguments:
        file_name: name of the file to be read.
        type_np: data type of the column.
    Outputs:
        entries: numpy matrix with the data (from the file) stored.
    """
    with open(file_name, 'r') as f_input:
        csv_input = csv.reader(f_input, delimiter=' ', skipinitialspace=True)
        entries = []
        for cols in csv_input:
            if type_np == 'float':
                entries.append(float(cols[0]))
            else:
                entries.append(int(cols[0]))
    return np.array(entries)


def get_hidden_layers(autoencoder):
    """
    Gets the hidden layers of a given autoencoder.
    Arguments:
        autoencoder: autoencoder as a stack of layers.
    Outputs:
        layers: list with the hidden layers.
    """
    # get the hidden layers from the encoder and the decoder
    layers_encoder = list(autoencoder.encoder.children())
    layers_decoder = list(autoencoder.decoder.children())

    # most of the layers are embedded in a nn.ModuleList(),
    # but we should consider as well the output layers
    # (linear and non-linear) from the encoder and decoder
    layers = list(layers_encoder[0]) + [layers_encoder[1]] \
             + list(layers_decoder[0]) + [layers_decoder[1]]

    return layers

# metrics to evaluate quality of the clustering
def get_purity(labels, predictions):
    """
     Gets the purity or accuracy of a clustering algorithm given the labels and predictions.
     Arguments:
         labels: true labels of the data.
         predictions: predicted labels of the data.
     Outputs:
         purity_score: score that represents the accuracy of a clustering algorithm.
     """

    #based on the code from https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py)
    assert len(labels) == len(predictions)
    d = max(np.max(labels), np.max(predictions)) + 1
    w = np.zeros((d,d))
    for i in range(len(predictions)):
        w[predictions[i], labels[i]] += 1

    ind = linear_assignment(w.max() - w) # optimal label mapping based on the Hungarian algorithm
    i, j = zip(*ind)
    purity_score = np.sum(w[i,j]) / len(predictions)

    return np.round(purity_score, 3)

def get_NMI(labels, predictions):
    """
     Computes the NMI (normalized mutual information) of a clustering algorithm given the labels and predictions.
     Arguments:
         labels: true labels of the data.
         predictions: predicted labels of the data.
     Outputs:
         NMI: score that represents the NMI of a clustering algorithm.
     """

    NMI = normalized_mutual_info_score(labels, predictions)
    return np.round(NMI, 3)

def get_ARI(labels, predictions):
    """
     Computes the ARI (adjusted Rand index) of a clustering algorithm given the labels and predictions.
     Arguments:
         labels: true labels of the data.
         predictions: predicted labels of the data.
     Outputs:
         ARI: score that represents the ARI of a clustering algorithm.
     """

    ARI = adjusted_rand_score (labels, predictions)
    return np.round(ARI, 3)