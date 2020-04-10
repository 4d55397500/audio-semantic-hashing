# test_training.py

import torch
import unittest

import training


class TestTraining(unittest.TestCase):

    def setUp(self):
        pass

    def test_loss_criterion(self):
        # cross entropy
        output = torch.randint(low=-32, high=32, size=(10, 1, 100)).float()
        target = torch.randint(low=-32, high=32, size=(10, 1, 100)).float()
        training.loss_criterion(output, target)


if __name__ == "__main__":
    unittest.main()

