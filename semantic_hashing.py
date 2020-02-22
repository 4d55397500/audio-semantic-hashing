# semantic_hashing.py
"""
    pytorch implementation of semantic hashing

"""
import torch


import constants


class SemanticHashing(torch.nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 initial_noise_std=constants.INITIAL_NOISE_STD,
                 noise_increment=constants.NOISE_MULTIPLICATIVE_INCREMENT):
        super(SemanticHashing, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.noise_sigma = initial_noise_std
        self.noise_increment = noise_increment
        self.i = 0

    def binary_encoding(self, x):
        threshold = torch.Tensor([0.5])
        z = torch.sigmoid(self.encoder.forward(x))
        return (z > threshold).float() * 1

    def encoded_entropy(self, x):
        binary_enc = self.binary_encoding(x)
        _, row_counts = torch.unique(binary_enc, return_counts=True, dim=0)
        probs = row_counts * 1. / torch.sum(row_counts)
        return -torch.sum(probs * torch.log(probs))

    def forward(self, x):
        enc_x = self.encoder.forward(x)
        z = enc_x + self.noise_sigma * torch.randn(size=enc_x.size())
        self.noise_sigma *= self.noise_increment
        return self.decoder(torch.sigmoid(z))


class DenseEncoder(torch.nn.Module):

    def __init__(self,
                 xdim=constants.WAV_CHUNK_SIZE,
                 zdim=constants.ENCODED_BITSEQ_LENGTH,
                 intermediate_dims=constants.INTERMEDIATE_LAYER_DIMS):
        super(DenseEncoder, self).__init__()
        dims = [xdim] + intermediate_dims + [zdim]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )

    def forward(self, x):
        current_x = x
        for i, l in enumerate(self.layers):
            current_x = l(current_x)  # intermediate activations?
        return current_x


class DenseDecoder(torch.nn.Module):

    def __init__(self,
                 xdim=constants.WAV_CHUNK_SIZE,
                 zdim=constants.ENCODED_BITSEQ_LENGTH,
                 intermediate_dims=constants.INTERMEDIATE_LAYER_DIMS):
        dims = ([xdim] + intermediate_dims + [zdim])[::-1]
        super(DenseDecoder, self).__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in
             range(len(dims) - 1)]
        )

    def forward(self, z):
        current_z = z
        for i, l in enumerate(self.layers):
            current_z = l(current_z)  # intermediate activations?
        return current_z


