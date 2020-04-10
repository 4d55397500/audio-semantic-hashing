import os
import unittest

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

    def test_chunks_to_numpy(self):
        x = audio_ops.chunks_dir_to_numpy(constants.LOCAL_CHUNK_FILEPATHS)
        num_chunks = len(os.listdir(constants.LOCAL_CHUNK_FILEPATHS))
        assert x.shape[0] == num_chunks
        assert x.shape[1] == constants.WAV_CHUNK_SIZE


if __name__ == "__main__":
    unittest.main()


