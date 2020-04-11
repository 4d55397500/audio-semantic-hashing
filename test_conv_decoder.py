# test_conv_decoder.py

import torch
import unittest

from conv_decoder import ConvDecoder


class TestConvDecoder(unittest.TestCase):

    def setUp(self):
        self.cd = ConvDecoder()
        self.z = torch.ones(10, 100)

    def test_forward(self):
        x = self.cd(self.z)
        assert x.shape == (10, 256, 1000)


if __name__ == "__main__":
    unittest.TestCase()
