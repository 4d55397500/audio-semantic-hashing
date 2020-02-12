import unittest

import audio_ops
import constants


class TestAudioOps(unittest.TestCase):

    def setUp(self):
        pass

    def test_chunk_audio(self):
        audio_ops.chunk_audio("./wavs/cello.wav", "./chunks",
                              constants.WAV_CHUNK_SIZE)


if __name__ == "__main__":
    unittest.main()
