import unittest

import api
import constants


class TestApi(unittest.TestCase):

    def setUp(self):
        self.app = api.app.test_client()
        self.remote_file_paths = \
            [url for urls in constants.SAMPLE_AUDIO.values() for url in urls]

    def tearDown(self):
        pass

    def test_add(self):
        response = self.app.post("/add",
                                 json={"filepaths": self.remote_file_paths})
        content = response.json
        assert 'local_filepaths' in content, 'missing response key'
        assert 'wavs/harpsi-cs.wav' in content['local_filepaths'], 'missing wav'
        assert 'wavs/cello.wav' in content['local_filepaths'], 'missing wav'

    def test_dataset_info(self):
        response = self.app.get("/datasetinfo")
        content = response.json
        assert 'num_wavs' in content, "'num wavs' key missing"
        assert 'num_chunks' in content, "'num chunks' key missing"
        assert content['num_wavs'] >= 0
        assert content['num_chunks'] >= 0

    def test_train(self):
        response = self.app.post("/train",
                                 json={"n_epochs": 100})
        content = response.json
        assert 'status' in content
        assert content['status'] == 'success'

    def test_index(self):
        response = self.app.post('/index')
        content = response.json
        assert 'status' in content
        assert content['status'] == 'success'

    def test_search(self):
        test_wav = "test_resources/test.wav"
        with open(test_wav, 'rb') as handle:
            data = dict(
                file=(handle, 'test.wav')
            )
            response = self.app.post('/search',
                                     data=data,
                                     content_type='multipart/form-data')


if __name__ == "__main__":
    unittest.main()


