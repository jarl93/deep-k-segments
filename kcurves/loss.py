# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from helpers import  pairwise_distances, pairwise_distances_segments, get_hidden_layers

# As long as PyTorch operations are employed the loss.backward should work

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

    :param h:
    :param alpha_:
    :param gamma_:
    :return:
    The implementaion can seem freaky, but it's because the implementation of the function torch.nn.KLDivLoss.
    Check out the documentation in https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """
    q_softmax = F.softmax(-1.0 * alpha_ * dist, dim=1)
    q_mean = torch.mean(q_softmax, dim = 0)
    loss_KL = torch.nn.KLDivLoss ()
    KL_last_layer_loss = gamma_* loss_KL(q_mean.log(), p_ref)

    return KL_last_layer_loss

def check_non_neagtive_loss(loss, name, x, x_reconstructed):
    """
    Checks that a given loss is non-negative.
    :param loss:
    :param name:
    :return:
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


    #----------------------------------Frozen code ------------------------------------------------#
    # elif type_loss == "entropy":
    #
    #     dist_entropy = F.softmax(-alpha_ * dist, dim=1) * F.log_softmax(-alpha_ * dist, dim=1)
    #     loss_entropy = -1.0 * beta_ * dist_entropy.sum()
    #     loss_entropy = check_non_neagtive_loss(loss_entropy, "loss_entropy")
    #     loss_batch += loss_entropy
    #     loss_KL = KL_loss(dist, alpha_, gamma_, p_ref)
    #     loss_KL = check_non_neagtive_loss(loss_KL, "loss_KL")
    #     loss_batch += loss_KL
    #     return loss_batch, loss_rec, loss_entropy, loss_KL


# ---------------------------------------------Frozen code -------------------------------------------------------------
# def KL_regularization(autoencoder, x, gamma_, rho_scalar):
#     """
#     Computes the KL divergence regularization loss by means of a Bernoulli distribution.
#     The hyperparameter rho is assumed to be the true average of the activation of a
#     neuron in each activation layer, rho_hat is the estimated average over the sample batch,
#     then, the KL divergence is computed between the two Bernoulli distributions parametrized by
#     rho and rho_hat.
#     :param autoencoder: auto-encoder in which the training is being run.
#     :param x: input of the auto-encoder.
#     :param gamma_: hyperparameter to scale the regularization term.
#     :param rho_scalar: hyperparameter to define the true Bernoulli distribution rho.
#     :return: KL divergence regularization loss.
#     """
#     layers = get_hidden_layers(autoencoder)
#     loss_KL = 0
#     for layer in layers:
#         # print("Shape x: {}".format(x.shape))
#         x = layer(x)
#         if isinstance(layer, nn.ReLU):
#             # compute the KL divergence
#             rho_hat = torch.mean(torch.sigmoid(x), 0)
#             rho = rho_scalar * torch.ones_like(rho_hat)
#             # print("rho: ", rho)
#             # print("rho_hat: ", rho_hat)
#             loss_KL += torch.sum(rho*torch.log(rho/rho_hat) + (1-rho)*torch.log((1-rho)/(1-rho_hat)))
#
#
#     # sys.exit("Stopping script!")
#
#     # scale by gamma
#     loss_KL *= gamma_
#
#     return loss_KL
#
# def entropy_regularization(encoder, h, beta_):
#     """
#     Computes the entropy regularization loss for the last layer of the encoder
#     by applying the softmax function to the last layer and then applying the entropy loss.
#     Arguments:
#         encoder: encoder in which the training is being run.
#         h: output of the encoder (latent variable).
#         beta_: hyperparameter to scale the regularization term.
#     Outputs:
#         loss_entropy: entropy regularization loss of the encoder (last layer).
#
#     """
#     # compute entropy loss for the last layer of the encoder
#     h_ent = None
#     if encoder.last_nn_layer_encoder_name == 'Identity':
#         h_ent = F.softmax(h, dim = 1) * F.log_softmax(h, dim = 1)
#         #TODO: add code for: h_ent = F.softmax(h, dim=1) * h
#     elif encoder.last_nn_layer_encoder == 'Softmax':
#         h_ent = h * torch.log(h)
#
#     loss_entropy = -1.0 * h_ent.sum()
#
#     rho_hat = F.softmax(h, dim = 1)
#     rho = 0.5 * torch.ones_like(rho_hat)
#     loss_KL = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
#
#     loss_entropy += loss_KL
#
#     # scale by lambda
#     loss_entropy *= beta_
#
#     return loss_entropy
#
# def loss_function(x, x_reconstructed, h, autoencoder, scalar_hyperparameters, regularization_types = None):
#     """
#     Computes the loss_function of the autoencoder.
#     Arguments:
#         x_reconstructed: reconstructed input.
#         x: input tensor.
#         h: output of the encoder (latent variable).
#         autoencoder: auto-encoder in which the training is being run.
#         scalar_hyperparameters: list of hyperparameters to scale the regularization terms, depending on
#             the regularization type. The description of the parameters is the following:
#             - lambda_: scalar for the  L1 regularization.
#             - beta_: scalar for the entropy regularization.
#             - gamma_: scalar for the KL divergence regularization.
#             - rho_: hyperparamter for the true Bernoulli distribution used for the KL-divergence regularization.
#         regularization_types: list of boolean values depending on whether the regularization is activated or not.
#             The regularization types are the following:
#             - reg_L1: boolean to activate the regularization L1.
#             - reg_KL: boolean to activate the regularization KL-divergence.
#             - reg_entropy: boolean to activate the entropy regularization.
#
#     Outputs:
#         loss_batch: loss of the batch depending on the type of the regularization used.
#
#     """
#
#     lambda_, beta_, gamma_, rho_ = scalar_hyperparameters
#     reg_L1, reg_KL, reg_entropy = regularization_types
#
#     # Compute the MSE loss between the input and the reconstruction
#     loss_MSE = nn.MSELoss()
#     loss_batch = loss_MSE(x, x_reconstructed)
#
#     if regularization_types is not None:
#         if reg_L1:
#             loss_L1 = L1_regularization(autoencoder, x, lambda_)
#             loss_batch += loss_L1
#         if reg_KL:
#             loss_KL = KL_regularization(autoencoder, x, gamma_, rho_)
#             loss_batch += loss_KL
#         if reg_entropy:
#             loss_entropy = entropy_regularization(autoencoder.encoder, h, beta_)
#             loss_batch += loss_entropy
#
#     return loss_batch




