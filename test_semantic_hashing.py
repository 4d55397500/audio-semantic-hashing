import unittest
from semantic_hashing import SemanticHashing
from conv_decoder import ConvDecoder
from conv_encoder import ConvEncoder
import constants

import torch


class TestSemanticHashing(unittest.TestCase):

    def setUp(self):
        de = ConvEncoder()
        dd = ConvDecoder()
        self.model = SemanticHashing(
            encoder=de,
            decoder=dd
        )

    def tearDown(self):
        pass

    def test_forward(self):
        x = torch.Tensor([[1.] * constants.WAV_CHUNK_SIZE,
                          [0.] * constants.WAV_CHUNK_SIZE]).unsqueeze(dim=1)
        num_channels = 1
        assert x.shape == (2, num_channels, constants.WAV_CHUNK_SIZE)
        x_pred = self.model(x)
        assert x_pred.shape == (2, num_channels, constants.WAV_CHUNK_SIZE)
        print(x_pred.shape)


    def testEncodedEntropy(self):
        x = torch.Tensor([[1.] * constants.WAV_CHUNK_SIZE,
                          [0.] * constants.WAV_CHUNK_SIZE]).unsqueeze(dim=1)
        actual = self.model.encoded_entropy(x)
        expected1 = torch.log(torch.Tensor([2.,]))
        expected2 = torch.Tensor([0.,])
        assert(torch.allclose(actual, expected1) or torch.allclose(actual, expected2))


if __name__ == "__main__":
    unittest.main()
