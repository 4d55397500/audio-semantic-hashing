import torch

import constants
from left_padded_conv import LeftPaddedConv


class ConvDecoder(torch.nn.Module):

    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.block_dilations = list(reversed([2 ** i for i in range(constants.NUM_CONV_LAYERS)]))
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(constants.ENCODED_BITSEQ_LENGTH,
                             constants.WAV_CHUNK_SIZE)] +
            [LeftPaddedConv(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                dilation=dilation
            ) for dilation in self.block_dilations]
        )

    def forward(self, z):
        current_z = z
        for i, l in enumerate(self.layers):
            current_z = l(current_z)  # intermediate activations?
        return current_z
