import unittest
import base64
import io

import local_index
import training
import custom_exceptions


class TestLocalIndex(unittest.TestCase):

    def setUp(self):
        training.train_pytorch(300, 1)

    def tearDown(self):
        pass

    def test_create_index(self):
        local_index.create_index()

    def test_run_search(self):
        sample_search_chunk = './chunks/test_0.wav'
        wav_bytes = open(sample_search_chunk, 'rb').read()
        try:
            results = local_index.run_search(wav_bytes)
        except custom_exceptions.IndexNotFoundException:
            local_index.create_index()
            results = local_index.run_search(wav_bytes)
        for k, v in results:
            assert v >= 0., 'negative distance in search result'
            try:
                wav_bytes = base64.b64decode(k)
                fp = io.BytesIO(wav_bytes)
                # test can read wav bytes
            except:
                self.fail("unable to decode base64 encoded key from string to bytes")

        # check base64 encoded key and value is >= 0.

if __name__ == "__main__":
    unittest.main()
