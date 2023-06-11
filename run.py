from nano_nerf import nerf
from nano_nerf import dataloader
from tqdm import tqdm
import torch
import torch.nn as nn
# from glob import glob
# from torchvision import transforms

def train_nerf(dataloader,epochs):
    model = nerf.NeRF()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for img, trans in tqdm(dataloader):
            optimizer.zero_grad()
            '''
            to implement the rest
            '''
    pass

if __name__ == "__main__":
    train_nerf(dataloader.dataloader,20)
