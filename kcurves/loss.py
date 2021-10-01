# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from helpers import  pairwise_distances, pairwise_distances_segments, get_hidden_layers

def L1_regularization(autoencoder, x, lambda_):
    """
    Computes the L1 regularization loss for the autoencoder (sparse autoencoder).
    Arguments:
        autoencoder: auto-encoder in which the training is being run.
        x: input tensor.
        lambda_ : hyperparameter to scale the regularization term.
    Outputs:
        loss_L1: L1 regularization loss for the autoencoder.

    """
    loss_L1 = 0
    layers = get_hidden_layers(autoencoder)

    for layer in layers:
        x = layer(x)
        if isinstance(layer, nn.ReLU): # consider just the activation layers
            loss_L1 += torch.mean(torch.abs(x))  # get the mean of the batch

    # scale by lambda
    loss_L1 *= lambda_

    return loss_L1

def KL_loss(dist, alpha_, gamma_, p_ref):
    """
    Performs the KL divergence loss between the the reference distribution p_ref and the softmax of the dist,
    which induces a distribution over dist.
    The implementation can seem freaky, but it's due to the implementation of the function torch.nn.KLDivLoss.
    Check out the documentation in https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    Arguments:
        dist: tensor with distances.
        alpha_: hyperparameter for the softmax.
        gamma_: hyperparameter for the loss.
        p_ref: reference distribution.
    Outputs:
        KL_last_layer_loss: KL divergence loss.
    """
    q_softmax = F.softmax(-1.0 * alpha_ * dist, dim=1)
    q_mean = torch.mean(q_softmax, dim = 0)
    loss_KL = torch.nn.KLDivLoss ()
    KL_last_layer_loss = gamma_* loss_KL(q_mean.log(), p_ref)

    return KL_last_layer_loss

def check_non_neagtive_loss(loss, name, x, x_reconstructed):
    """
    Checks that a given loss is non-negative.
    Arguments:
        loss: loss to be checked.
        name: name of the loss.
        x: data input.
        x_reconstructed: data input reconstructed.
    Outputs:
        loss: same loss if the sanity check was passed.
    """
    eps = -1e-3
    if loss.item() <= eps:
        print(name + " = {}".format(loss))
        raise ValueError(name + " cannot be negative!")
    elif torch.isnan(loss):
        print(name + " = {}".format(loss))
        print("x = ", x)
        print("x_rec = ", x_reconstructed)
        raise ValueError(name + " is nan!")
    else:
        return loss

def loss_function(model, x, x_reconstructed, h, rep, alpha_, beta_, gamma_, type_rep, type_loss):
    """
    Main loss function of the model.
    Arguments:
        model: model where the loss is applied.
        x: data input.
        x_reconstructed: data input reconstructed.
        h: latent variable.
        rep: current segments (of the training).
        alpha_: hyperparameter for the softmax.
        beta_: hyperparamter for the relaxation of the distance to the representatives.
        gamma_: hyperparameter for the loss of the length of the representatives.
        type_rep: type of representatives, it could be points or representatives.
        type_loss: type of loss to be applied.
    Outputs:
        loss_batch: total loss of the batch.
        loss_rec: reconstruction loss.
        loss_dist_log: loss of the relaxation of the distance to the representatives.
    """
    # Compute the MSE loss between the input and the reconstruction
    loss_MSE = nn.MSELoss()
    loss_rec = loss_MSE(x, x_reconstructed)
    loss_rec = check_non_neagtive_loss(loss_rec, "loss_rec", x, x_reconstructed)
    loss_batch = torch.zeros_like(loss_rec)
    loss_batch += loss_rec

    dim = h.shape[1]
    if type_rep == "points":
        # compute the distance matrix between the batch of points and the representatives
        dist = pairwise_distances(h, rep)

    elif type_rep == "segments":
        dist = pairwise_distances_segments(h, rep)
        diff = rep[:, dim:] - rep[:, :dim]
        loss_length = gamma_ * (1/diff.shape[0]) * torch.sum(diff ** 2)
        loss_batch += loss_length
    elif type_rep == "axes":
        dist = torch.abs(h)

    # compute the minimum for numerical stability in the softmax function
    min_dist, _ = torch.min(dist, dim=1)
    min_dist = min_dist.view(-1,1).repeat(1,dist.shape[1])

    if type_loss == "dist":
        # relaxation of the Indicator function by means of the softmax function
        I_relaxed = F.softmax(-1 * alpha_ * (dist - min_dist), dim = 1)
        # compute the loss by multiplying the Indicator function and the distance
        loss_dist = beta_ * torch.mean(torch.sum(I_relaxed * dist, dim = 1))
        loss_dist = check_non_neagtive_loss(loss_dist, "loss_dist", x, x_reconstructed)
        loss_batch += loss_dist
        return loss_batch, loss_rec, loss_dist
    elif type_loss == "log_dist":
        # relaxation of the Indicator function by means of the softmax function
        I_relaxed = F.softmax(-1 * alpha_ * (dist - min_dist), dim=1)
        # compute the loss by multiplying the Indicator function and the distance
        eps = 0.1 * torch.ones_like(dist)
        loss_dist_log = beta_ * torch.mean(torch.sum(I_relaxed * torch.log(dist+eps), dim = 1))
        loss_batch += loss_dist_log
        return loss_batch, loss_rec, loss_dist_log




