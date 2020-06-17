import os
import unittest
import torch

import audio_ops
import ops
import constants


class TestAudioOps(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        ops.ensure_dirs()
        self.test_wav_path = "./test_resources/test.wav"

    @classmethod
    def tearDownClass(self):
        ops.clean_dirs()

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
            for ve in v:
                for e in ve.data.cpu().numpy():
                    assert e in allowed_values, \
                    f"mu quantization out of range: {e} found"

    def test_chunks_to_numpy(self):
        x = audio_ops.chunks_dir_to_numpy(constants.LOCAL_CHUNK_FILEPATHS)
        num_chunks = len(os.listdir(constants.LOCAL_CHUNK_FILEPATHS))
        assert x.shape == (num_chunks, 256,constants.WAV_CHUNK_SIZE)


if __name__ == "__main__":
    unittest.main()


