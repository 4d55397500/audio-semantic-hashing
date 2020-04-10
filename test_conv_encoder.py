# test_conv_encoder.py

import torch
import unittest

import constants
from conv_encoder import ConvEncoder


class TestConvEncoder(unittest.TestCase):

    def setUp(self):
        self.ce = ConvEncoder()

    def test_forward(self):

        x = torch.Tensor([[1.] * constants.WAV_CHUNK_SIZE,
                          [0.] * constants.WAV_CHUNK_SIZE]).unsqueeze(dim=1)
        assert x.shape == (2, 1, constants.WAV_CHUNK_SIZE)
        y = self.ce(x)
        assert y.shape == (2, 1, constants.ENCODED_BITSEQ_LENGTH)


if __name__ == "__main__":
    unittest.TestCase()
