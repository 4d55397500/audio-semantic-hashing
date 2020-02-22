import unittest
from semantic_hashing import SemanticHashing, DenseDecoder, DenseEncoder
import constants

import torch


class TestSemanticHashing(unittest.TestCase):

    def setUp(self):
        de = DenseEncoder()
        dd = DenseDecoder()
        self.model = SemanticHashing(
            encoder=de,
            decoder=dd
        )

    def tearDown(self):
        pass

    def testEncodedEntropy(self):
        x = torch.Tensor([[1.] * constants.WAV_CHUNK_SIZE,
                          [0.] * constants.WAV_CHUNK_SIZE])
        actual = self.model.encoded_entropy(x)
        expected = torch.log(torch.Tensor([2.,]))
        assert(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
