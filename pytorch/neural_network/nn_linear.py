# 2023.4.6

import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Baihua(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.Linear1(input)
        return output
    
baihua = Baihua()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))#stretch
    output = torch.flatten(imgs)
    print(output.shape)

    output = baihua(output)
    print(output.shape)


