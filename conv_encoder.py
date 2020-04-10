import torch

import constants
from left_padded_conv import LeftPaddedConv


class ConvEncoder(torch.nn.Module):

    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.block_dilations = [2 ** i for i in range(constants.NUM_CONV_LAYERS)]
        self.layers = torch.nn.ModuleList(
            [LeftPaddedConv(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                dilation=dilation
            ) for dilation in self.block_dilations]
            + [torch.nn.Linear(constants.WAV_CHUNK_SIZE, constants.ENCODED_BITSEQ_LENGTH)]
        )

    def forward(self, x):
        current_x = x
        for i, l in enumerate(self.layers):
            current_x = l(current_x)  # intermediate activations?
        return current_x

