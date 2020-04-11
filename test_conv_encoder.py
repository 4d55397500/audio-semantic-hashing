# test_conv_encoder.py

import torch
import unittest

from conv_encoder import ConvEncoder


class TestConvEncoder(unittest.TestCase):

    def setUp(self):
        self.ce = ConvEncoder()
        self.x = torch.ones(10, 256, 1000)

    def test_forward(self):
        z = self.ce(self.x)
        assert z.shape == (10, 100)


if __name__ == "__main__":
    unittest.TestCase()
