# test_training.py

import constants
import ops
import torch
import unittest

import audio_ops
import training



def chunk_test_resources():
    audio_ops.chunk_write_audio(constants.TEST_WAV_PATH)


class TestTraining(unittest.TestCase):

    def setUp(self):
        ops.ensure_dirs()
        chunk_test_resources()
        self.mock_target = torch.nn.functional.one_hot(
            torch.randint(low=0,
                      high=256,
                      size=(10, 1000)),
            256).permute(0, 2, 1)
        assert self.mock_target.shape == (10, 256, 1000)
        assert torch.max(torch.sum(self.mock_target, dim=1) == 1)
        self.mock_output = torch.randn((10, 256, 1000))

    def tearDown(self):
        ops.clean_dirs()

    def test_loss_criterion(self):
        loss = training.loss_criterion(self.mock_output,
                                       self.mock_target)

    def test_train_pytorch(self):
        training.train_pytorch(batch_size=constants.TRAINING_BATCH_SIZE,
                               n_epochs=1)



if __name__ == "__main__":
    unittest.main()

