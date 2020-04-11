import torch

import constants
from left_padded_conv import LeftPaddedConv


class ConvEncoder(torch.nn.Module):

    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.block_dilations = [2 ** i for i in range(constants.NUM_CONV_LAYERS)]

        self.layers = torch.nn.ModuleList(
            [LeftPaddedConv(
                in_channels=256 if i == 0 else constants.INTERMEDIATE_CHANNEL_DIM,
                out_channels=1 if i == (constants.NUM_CONV_LAYERS - 1) else constants.INTERMEDIATE_CHANNEL_DIM,
                kernel_size=2,
                dilation=dilation
            ) for i, dilation in enumerate(self.block_dilations)]
            + [torch.nn.Linear(constants.WAV_CHUNK_SIZE, constants.ENCODED_BITSEQ_LENGTH)]
        )

    def forward(self, x):
        current_x = x
        for i, l in enumerate(self.layers):
            current_x = l(current_x)  # intermediate activations?
        return current_x.squeeze(dim=1)

