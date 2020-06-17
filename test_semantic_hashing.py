import unittest
from semantic_hashing import SemanticHashing, row_entropy
from conv_decoder import ConvDecoder
from conv_encoder import ConvEncoder

import torch


class TestSemanticHashing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = SemanticHashing(
            encoder=ConvEncoder(),
            decoder=ConvDecoder(),
        )
        cls.x = torch.ones(10, 256, 1000)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_forward(self):
        x_out = self.model(self.x)
        assert x_out.shape == (10, 256, 1000)

    def test_encoded_entropy(self):
        enc_ent = self.model.encoded_entropy(self.x)
        assert enc_ent >= 0.

    def test_binary_encoding(self):
        binary_enc = self.model.binary_encoding(self.x)
        assert binary_enc.shape == (10, 100)

    def test_row_entropy(self):
        rand_x = torch.randint(low=0,
                               high=2,
                               size=(200, 100))
        ent = row_entropy(rand_x)
        assert ent > 0.


if __name__ == "__main__":
    unittest.main()
