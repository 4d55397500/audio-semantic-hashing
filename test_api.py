import base64
import io
import scipy
import os
import unittest
import shutil


import api
import ops
import constants
import test_training
unittest.TestLoader.sortTestMethodsUsing = None


class TestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ops.ensure_dirs()
        test_training.chunk_test_resources()
        cls.app = api.app.test_client()
        #self.remote_file_paths = \
         #   [url for urls in constants.SAMPLE_AUDIO.values() for url in urls]

    @classmethod
    def tearDownClass(cls):
        ops.clean_dirs()

    @unittest.skip("network connection remote build")
    def test_add(self):
        response = self.app.post("/add",
                                 json={"filepaths": []})#self.remote_file_paths})
        content = response.json
        assert 'local_filepaths' in content, 'missing response key'


    def test_dataset_info(self):
        response = self.app.get("/datasetinfo")
        content = response.json
        assert 'num_wavs' in content, "'num wavs' key missing"
        assert 'num_chunks' in content, "'num chunks' key missing"
        assert content['num_wavs'] >= 0
        assert content['num_chunks'] >= 0

    @unittest.skip("no need training already tested in test_training")
    def test_train(self):
        response = self.app.post("/train",
                                 json={"n_epochs": 1})
        content = response.json
        assert 'status' in content
        assert content['status'] == 'success'

    def test_index(self):
        response = self.app.post('/index')
        content = response.json
        assert 'status' in content
        assert content['status'] == 'success' \
            or content['status'] == 'model not found'

    def test_search(self):
        test_wav = "test_resources/test.wav"
        with open(test_wav, 'rb') as handle:
            data = dict(
                file=(handle, 'test.wav')
            )
            response = self.app.post('/search',
                                     data=data,
                                     content_type='multipart/form-data')
            content = response.json
            assert 'results' in content
            for wav_str, distance in content['results']:
                wav_bytes = base64.b64decode(wav_str)
                fp = io.BytesIO(wav_bytes)
                # test can read wav bytes
                rate, np_audio = scipy.io.wavfile.read(fp)
                #assert rate == 48000, f"incorrect rate {rate}"
                assert distance >= 1.0, "invalid distance"


if __name__ == "__main__":
    unittest.main()


