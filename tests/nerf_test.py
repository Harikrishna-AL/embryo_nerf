import unittest
import torch
import torch.nn as nn
from nano_nerf import nerf


class TestNeRF(unittest.TestCase):
    def setUp(self):
        self.model = nerf.NeRF()

    def test_forward_pass(self):
        input = torch.randn(1, 3, 100, 100)
        output = self.model(input)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (3, 100, 100))

    def test_NeRF(self):
        self.assertIsInstance(self.model, nn.Module)


if __name__ == "__main__":
    unittest.main()
