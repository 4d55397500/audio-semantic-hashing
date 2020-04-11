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
        binary_enc = self.binary_encoding(x).int()
        _, row_counts = torch.unique(binary_enc, return_counts=True, dim=0)
        probs = row_counts * 1. / torch.sum(row_counts)
        return -torch.sum(probs * torch.log(probs))

    def forward(self, x):
        enc_x = self.encoder.forward(x)
        z = enc_x + self.noise_sigma * torch.randn(size=enc_x.size())
        self.noise_sigma *= self.noise_increment
        return self.decoder(torch.sigmoid(z))


def row_entropy(tensor):
    _, row_counts = torch.unique(tensor, return_counts=True, dim=0)
    probs = row_counts * 1. / torch.sum(row_counts)
    return -torch.sum(probs * torch.log(probs))
