from nano_nerf import nerf
from tqdm import tqdm
import torch
import torch.nn as nn
def train_nerf(dataloader,epochs):
    model = nerf.NeRF()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        '''
        to implement the rest
        '''
    pass

if __name__ == "__main__":
    train_nerf()
