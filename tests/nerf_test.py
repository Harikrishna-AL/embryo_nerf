import unittest
import torch
import torch.nn as nn
from nano_nerf import nerf, train
from nano_nerf.utils import pos_encoding


class TestNeRF(unittest.TestCase):
    def setUp(self):
        self.model = nerf.NeRF()
        self.train_loop = train.train_iter

    def test_forward_pass(self):
        input = torch.randn(1, 3, 100, 100)
        trans_input = torch.randn(1, 1, 4, 4)
        output = self.train_loop(
            input.shape[-1],
            input.shape[-2],
            113,
            trans_input,
            2,
            6,
            32,
            lambda x: pos_encoding(x),
            self.model,
        )
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, torch.Size([100, 100, 3]))

    def test_NeRF(self):
        self.assertIsInstance(self.model, nn.Module)


if __name__ == "__main__":
    unittest.main()
