# test_training.py

import torch
import unittest

import training


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.mock_target = torch.nn.functional.one_hot(
            torch.randint(low=0,
                      high=256,
                      size=(10, 1000)),
            256).permute(0, 2, 1)
        assert self.mock_target.shape == (10, 256, 1000)
        assert torch.max(torch.sum(self.mock_target, dim=1) == 1)
        self.mock_output = torch.randn((10, 256, 1000))

    def test_loss_criterion(self):
        loss = training.loss_criterion(self.mock_output,
                                       self.mock_target)


if __name__ == "__main__":
    unittest.main()

