# 2023.4.6

import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
# torch.reshape(input, (-1,1,2,2))

dataset = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Baihua(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = ReLU(inplace=False)
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output
    
baihua = Baihua()
# output = baihua(input)
# print(output)
# tensor([[1., 0.],
#         [0., 3.]])

writer = SummaryWriter('log')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("relu_input", imgs, step)
    output = baihua(imgs)
    writer.add_images("relu_output", output, step)
    step = step + 1

writer.close()