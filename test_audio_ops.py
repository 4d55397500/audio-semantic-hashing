import os
import unittest
import torch

import audio_ops
import constants


class TestAudioOps(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_chunk_write_audio(self):
        audio_ops.chunk_write_audio("./test_resources/test.wav",
                                    "./chunks",
                                    constants.WAV_CHUNK_SIZE)

    def test_mu_transform(self):
        x = torch.randn(2, 1, 100)
        mu_x = audio_ops.mu_transform(x,
                                      quantization_channels=256)
        print(mu_x)

    def test_chunks_to_numpy(self):
        x = audio_ops.chunks_dir_to_numpy(constants.LOCAL_CHUNK_FILEPATHS)
        num_chunks = len(os.listdir(constants.LOCAL_CHUNK_FILEPATHS))
        assert x.shape[0] == num_chunks
        assert x.shape[1] == constants.WAV_CHUNK_SIZE


if __name__ == "__main__":
    unittest.main()


