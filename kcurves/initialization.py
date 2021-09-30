# libraries
from helpers import load_config, pairwise_distances_segments
from model import AE
import numpy as np
import torch
from sklearn.decomposition import PCA
from constants import DEVICE
from helpers import get_hidden_layers
from numpy import linalg as LA
from clustering import k_means, k_segments
def init_model(cfg_path, verbose=True):
    """
    Add documentation
    :param encoder_layer_sizes:
    :param decoder_layer_sizes:
    :param input_dim:
    :param latent_dim:
    :param verbose:
    :return:
    """
    cfg_file = load_config(cfg_path)
    encoder_layer_sizes = cfg_file["model"]["encoder"]["layer_sizes"]
    decoder_layer_sizes = cfg_file["model"]["decoder"]["layer_sizes"]
    input_dim = cfg_file["model"]["input_dim"]
    latent_dim = cfg_file["model"]["latent_dim"]
    device = cfg_file["model"]["device"]
    space_init = cfg_file["train"]["space_init"]
    init_strategy = cfg_file["train"]["init_strategy"]
    scheme_train = cfg_file["train"]["scheme_train"]
    type_rep = cfg_file["train"]["type_rep"]
    type_loss = cfg_file["train"]["type_loss"]
    X_train = np.load(cfg_file["data"]["train"] + "X_train.npy")

    if verbose:
        print("Initialization of the model...")
        print("Scheme train: {}".format(scheme_train))
        print("Initialization strategy: {}".format(init_strategy))
        print("Space init: {}".format(space_init))
        print("Type loss: {}".format(type_loss))

    # get the representative according to the type of initialization
    rep_init = get_rep(cfg_path, X_train)

    # Define the model as an autoencoder
    model = AE(input_dim=input_dim, latent_dim = latent_dim, encoder_layer_sizes = encoder_layer_sizes,
               decoder_layer_sizes = decoder_layer_sizes, rep_init = rep_init,
               rep_type = type_rep, space_init = space_init)

    model = xavier_initialization(model)

    model = model.to(device)

    if verbose:
        print("Model: ", model)

    return model

def get_rep(cfg_path, X):

    cfg_file = load_config(cfg_path)
    input_dim = cfg_file["model"]["input_dim"]
    latent_dim = cfg_file["model"]["latent_dim"]
    K = cfg_file["data"]["num_classes"]
    init_strategy = cfg_file["train"]["init_strategy"]
    min_init = cfg_file["train"]["min_init"]
    diff_init = cfg_file["train"]["diff_init"]
    space_init = cfg_file["train"]["space_init"]
    fixed_length_start = cfg_file["train"]["fixed_length_start"]
    percentage_K = cfg_file["train"]["percentage_K"]
    type_rep = cfg_file["train"]["type_rep"]
    num_classes = cfg_file["data"]["num_classes"]


    #TODO: Add k_means++ and k-means for the case of points
    if type_rep == "points":
        if init_strategy == "random":
            if space_init == "input":
                rep_init = min_init + diff_init * np.random.rand(K, input_dim)
            elif space_init == "latent":
                rep_init = min_init + diff_init * np.random.rand(K, latent_dim)

        elif init_strategy == "forgy":
            rep_init = forgy_initialization(X, K)

    elif type_rep == "segments":
        if init_strategy == "fixed_length_random":
            dim = input_dim
            if space_init == "latent":
                dim = latent_dim
            rep_init = fixed_length_initialization(min_init, diff_init, fixed_length_start , K, dim)

        elif init_strategy == "forgy_PCA":
            centers_ = forgy_initialization(X, K)
            rep_init = get_segments_from_centers(centers_, X, K, percentage_K)

        elif init_strategy == "kmeans++_PCA":
            rep_init = kmeans_plusplus_PCA(X, K, percentage_K)

        elif init_strategy == "kmeans_PCA":
            rep_init = k_means_PCA(X, K, percentage_K)

        elif init_strategy == "ksegments":
            segments_init = k_means_PCA(X, K, percentage_K)
            rep_init, _ = k_segments(X, segments_init, K)

    # check whether rep_init is a Tensor or not
    if type(rep_init) == torch.Tensor:
        rep_init = rep_init.cpu().detach().numpy()

    return rep_init


def xavier_initialization(model):
    layers = get_hidden_layers(model)
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    return model

def fixed_length_initialization(min_init, diff_init, fixed_length, K, dim):

    s1 = min_init + diff_init * np.random.rand(K, dim)
    s_dir = min_init + diff_init * np.random.rand(K, dim)
    u_dir = s_dir/LA.norm(s_dir)
    s2 = s1 + fixed_length * u_dir
    rep_init = np.hstack((s1, s2))

    return rep_init


def forgy_initialization(X, K):
    N = X.shape[0]
    idx = np.random.choice(N, K)
    centers_init = X[idx]
    return centers_init

def get_segment_PCA(neigh, center):
    if torch.isnan(neigh).any():
        raise ValueError(" neigh has  nan values!")

    neigh = neigh.cpu().detach().numpy()
    center = center.cpu().detach().numpy()
    pca = PCA(n_components=1)
    pca.fit(neigh)
    c_std = 1.5
    length_s = c_std * np.sqrt(pca.explained_variance_[0])
    v_dir = pca.components_[0]
    s1 = center - length_s * v_dir
    s2 = center + length_s *v_dir
    s1 = torch.from_numpy(s1).to(DEVICE)
    s2 = torch.from_numpy(s2).to(DEVICE)
    return s1, s2

def kmeans_plusplus_PCA(X, K, percentage_K):

    N = X.shape[0]
    X = torch.from_numpy(X).to(DEVICE)
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)
    eps = 1e-6

    for i in range(K):
        if i == 0:
            idx = np.random.choice(N)
            center_i = X[idx]
            distance_center_i = torch.sum((X - center_i) ** 2, dim = 1)
            distance_sorted_i = torch.argsort(distance_center_i)
            neigh_i = X[distance_sorted_i[:scaled_length_k]]
            s1_i, s2_i = get_segment_PCA(neigh_i, center_i)
            s = torch.cat((s1_i, s2_i), 0)
            s = s.reshape(1, -1)
        else:
            dist_to_s = pairwise_distances_segments(X, s)
            dist_min, idx_min = torch.min(dist_to_s, 1)
            prob_X_torch = dist_min / (torch.sum(dist_min) + eps)
            prob_X = prob_X_torch.cpu().detach().numpy()
            idx_i = np.random.choice(N, p = prob_X)
            center_i = X[idx_i]
            distance_center_i = torch.sum((X - center_i) ** 2, dim = 1)
            distance_sorted_i = torch.argsort(distance_center_i)
            neigh_i =  X[distance_sorted_i[:scaled_length_k]]
            s1_i, s2_i = get_segment_PCA(neigh_i, center_i)
            s12_i = torch.cat((s1_i, s2_i), 0)
            s12_i = s12_i.reshape(1, -1)
            s = torch.cat((s, s12_i), 0)

        print("shape of s", s.shape)

    return s

def k_means_PCA(X, K, percentage_K):

    # get the centroids by means of k-means
    centers_, _ = k_means(X = X , centers_init='k-means++', n_clusters = K)
    s = get_segments_from_centers(centers_, X, K, percentage_K)
    print("shape of s", s.shape)
    return s


def get_segments_from_centers(centers_, X, K, percentage_K):

    centers = torch.from_numpy(centers_).to(DEVICE)
    N = X.shape[0]
    X = torch.from_numpy(X).to(DEVICE)
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)

    for i in range(K):
        center_i = centers[i]
        distance_center_i = torch.sum((X - center_i) ** 2, dim = 1)
        distance_sorted_i = torch.argsort(distance_center_i)
        neigh_i = X[distance_sorted_i[:scaled_length_k]]
        s1_i, s2_i = get_segment_PCA(neigh_i, center_i)
        s12_i = torch.cat((s1_i, s2_i), 0)
        s12_i = s12_i.reshape(1, -1)
        if i == 0:
            s = s12_i
        else:
            s = torch.cat((s, s12_i), 0)

    return s

