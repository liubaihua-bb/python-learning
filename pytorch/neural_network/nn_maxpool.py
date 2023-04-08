# 2023.4.6
# maxpool: default stride = kernelsize

import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision 

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1,],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)# the number of the [ is the dimension of the matrix

# input = torch.reshape(input, (-1,1,5,5))

dataset = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Baihua(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

baihua = Baihua()
# output = baihua(input)
# print(output)
# tensor([[[[2., 3.],
#           [5., 1.]]]])

writer = SummaryWriter('log')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("maxpool_input", imgs, step)
    output = baihua(imgs)
    writer.add_images("maxpool_output", output, step)
    step = step + 1

writer.close()