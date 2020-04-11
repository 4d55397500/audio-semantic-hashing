import torch

import constants
from left_padded_conv import LeftPaddedConv


class ConvDecoder(torch.nn.Module):

    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.block_dilations = list(
            reversed([2 ** i for i in range(constants.NUM_CONV_LAYERS)]))
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(constants.ENCODED_BITSEQ_LENGTH,
                             constants.WAV_CHUNK_SIZE)] +
            [LeftPaddedConv(
                in_channels=1 if i == 0 else constants.INTERMEDIATE_CHANNEL_DIM,
                out_channels=256 if i == (constants.NUM_CONV_LAYERS - 1) else constants.INTERMEDIATE_CHANNEL_DIM,
                kernel_size=2,
                dilation=dilation
            ) for i, dilation in enumerate(self.block_dilations)]
        )

    def forward(self, z):
        current_z = z.unsqueeze(dim=1)
        for i, l in enumerate(self.layers):
            current_z = l(current_z)  # intermediate activations?
        return current_z
