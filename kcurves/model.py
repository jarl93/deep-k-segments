# libraries
import torch.nn as nn
import torch
from constants import DEVICE

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layer_sizes, decoder_layer_sizes,
                 rep_init, rep_type, space_init):
        """
        Arguments:
            input_dim (int): dimension of the input.
            encoder_layer_sizes (list[int]): list with the sizes of the encoder layers.
            decoder_layer_sizes (list[int]): list with the sizes of the decoder layers.
            latent_dim (int): dimension of latent space/bottleneck.
            last_nn_layer_encoder (string): last non-linear layer of the encoder,
                                            the output will be the latent variable.
            last_nn_layer_decoder (string): last non-linear layer of the decoder,
                                            the output will be the reconstruction.
            rep_init (numpy matrix): matrix with the initial representatives.
        """
        super(AE, self).__init__()

        # the output dim is the same as the input dim
        self.input_dim = input_dim

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim)

        self.rep_type = rep_type


        # get representatives in the latent space depending on the space of init
        rep_ = torch.tensor(rep_init).type(torch.FloatTensor)


        # initialize the representatives
        if rep_type == "points":
            if rep_.shape[1] > latent_dim:
                rep_latent = self.get_rep(rep_)
            else:
                rep_latent = rep_
            self.rep = nn.Parameter(rep_latent)
        elif rep_type == "segments":
            if rep_.shape[1] > 2 * latent_dim:
                rep_latent = self.get_rep(rep_)
            else:
                rep_latent = rep_
            self.set_rep(rep_latent)

    def forward(self, x):
        """
        Forward Process of the whole AE.
        Arguments:
            x: tensor of dimension (batch_size, input_dim).
        Outputs:
            x_reconstructed: reconstructed input (output of the auto-encoder).
                Same dimension as x (batch_size, input_dim).
        """

        # Map the input to the latent space (encoding)
        z = self.encoder(x)

        # Map the latent variable to the input/output space (decoding),
        # i.e., get the reconstruction from the latent space
        x_reconstructed = self.decoder(z)

        return x_reconstructed

    def get_rep(self, s):
        if self.rep_type == "points":
            s_latent = self.encoder(s)
        elif self.rep_type == "segments":
            s1_ = s[:, :self.input_dim]
            s2_ = s[:, self.input_dim:]
            s1_latent = self.encoder(s1_)
            s2_latent = self.encoder(s2_)
            s_latent = torch.cat((s1_latent, s2_latent), 1)

        return s_latent
    def set_rep (self, rep):
        self.rep = nn.Parameter(rep)

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim):
        super(Encoder, self).__init__()
        """
        Arguments:
            layer_sizes (list[int]): list of sizes of the linear layers of the encoder.
            latent_dim (int): dimension of latent space, i.e. dimension out output of the encoder.
            last_nn_layer_encoder (string): last non-linear layer of the encoder, 
                                            the output will be the latent variable.
        """

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
            self.hidden.append(nn.ReLU())

        # Output layer from the encoder
        self.out = nn.Linear(layer_sizes[-1], latent_dim)

        # code to fix the encoder
        # with torch.no_grad():
        #     self.out.weight.copy_(torch.eye(latent_dim))



    def forward(self, x):
        """
        Makes the forward for the Encoder.
        Arguments:
            x: input of the auto-encoder, tensor of dimension (batch_size, 28*28).

        Outputs:
            z: latent variable, tensor of dimension (batch_size, latent_dim).
        """

        # Do the forward for the hidden layers
        for layer in self.hidden:
            x = layer(x)

        # Do the forward for the output layer
        z = self.out(x)

        return z

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim):
        super(Decoder, self).__init__()
        """
        Arguments:
            layer_sizes (list[int]): list of sizes of the linear layers of the decoder.
            latent_dim (int): dimension of latent space, i.e. dimension out input of the decoder.
            last_nn_layer_decoder (string): last non-linear layer of the decoder,
                                            the output will be the reconstructed input.

        """
        # Hidden layers
        self.hidden = nn.ModuleList()

        if len(layer_sizes) > 1:
            # Append the first layer of the decoder
            self.hidden.append(nn.Linear(latent_dim, layer_sizes[0]))
            self.hidden.append(nn.ReLU())
            #self.hidden.append(nn.Sigmoid())

            # Append the layers in between
            for k in range(len(layer_sizes) - 2):
                self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
                self.hidden.append(nn.ReLU())
                #self.hidden.append(nn.Sigmoid())

            # Append the last layer which considers the last element
            # from the list of layer_sizes, this could be done in the for loop
            # but it's done that way for the sake of neatness
            self.out = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        else:
            self.out = nn.Linear(latent_dim, layer_sizes[-1])
            # code to fix the decoder
            # with torch.no_grad():
            #     self.out.weight.copy_(torch.eye(latent_dim))

    def forward(self, z):
        """
        Makes the forward for the Decoder.
        Arguments:
            z: tensor of dimension (batch_size, latent_dim).
        Outputs:
            x: reconstructed input from the latent space of dimension (batch_size, input_dim).
        """
        # Do the feedforward for the hidden layers
        x = z
        for layer in self.hidden:
            x = layer(x)

        x = self.out(x)

        return x