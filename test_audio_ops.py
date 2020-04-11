import os
import unittest
import torch

import audio_ops
import constants

REAL_SAMPLE_CHUNK = 'waves_yesno/0_1_0_0_0_1_1_0.wav'


class TestAudioOps(unittest.TestCase):

    def setUp(self):
        self.test_wav_path = "./test_resources/test.wav"

    def tearDown(self):
        pass

    def test_chunk_write_audio(self):
        audio_ops.chunk_write_audio(self.test_wav_path,
                                    "./chunks",
                                    constants.WAV_CHUNK_SIZE)

    def test_mu_transform(self):
        x = torch.randn(2, 100) * 1013
        mu_x = audio_ops.mu_transform(x,
                                      quantization_channels=256)
        allowed_values = [float(i) for i in range(256)]
        for v in mu_x.squeeze():
            for e in v:
                assert e in allowed_values, \
                    f"mu quantization out of range: {e} found"

    def test_chunks_to_numpy(self):
        x = audio_ops.chunks_dir_to_numpy(constants.LOCAL_CHUNK_FILEPATHS)
        num_chunks = len(os.listdir(constants.LOCAL_CHUNK_FILEPATHS))
        assert x.shape == (num_chunks, 256,constants.WAV_CHUNK_SIZE)


if __name__ == "__main__":
    unittest.main()


