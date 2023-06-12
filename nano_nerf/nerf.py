import torch.nn as nn


class NeRF(nn.Module):
    def __init__(self, out_channel=128, num_encoding_functions=6):
        super(NeRF, self).__init__()
        self.layer1 = nn.Linear(3 + 3 * 2 * num_encoding_functions, out_channel)
        self.layer2 = nn.Linear(out_channel, out_channel)
        self.layer3 = nn.Linear(out_channel, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)
        return out
