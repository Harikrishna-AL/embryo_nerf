from nano_nerf import nerf
from nano_nerf import dataloader
from tqdm import tqdm
import torch
import torch.nn as nn
from nano_nerf import train
from nano_nerf.utils import pos_encoding

# from glob import glob
# from torchvision import transforms

encode = lambda x: pos_encoding(x)
depth_samples_per_ray = 32
focal = 113


def train_nerf(dataloader, epochs):
    model = nerf.NeRF()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for img, trans in tqdm(dataloader):
            optimizer.zero_grad()
            pred = train.train_iter(
                img.shape[-1],
                img.shape[-2],
                focal,
                trans,
                2,
                6,
                depth_samples_per_ray,
                encode,
                model,
            )
            img = img.reshape([100,100,3])
            loss = criterion(img.double(), pred.double())
            loss.backward()
            optimizer.step()
    pass


if __name__ == "__main__":
    train_nerf(dataloader.dataloader, 20)
