# test_conv_decoder.py

import torch
import unittest

import constants
from conv_decoder import ConvDecoder


class TestConvDecoder(unittest.TestCase):

    def setUp(self):
        self.cd = ConvDecoder()

    def test_forward(self):
        x = torch.Tensor([[1.] * constants.ENCODED_BITSEQ_LENGTH,
                          [0.] * constants.ENCODED_BITSEQ_LENGTH]).unsqueeze(dim=1)
        assert x.shape == (2, 1, constants.ENCODED_BITSEQ_LENGTH)
        y = self.cd(x)
        assert y.shape == (2, 1, constants.WAV_CHUNK_SIZE)



if __name__ == "__main__":
    unittest.TestCase()
