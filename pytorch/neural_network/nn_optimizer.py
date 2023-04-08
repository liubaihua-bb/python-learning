# 2023.4.6

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)

class Baihua(nn.Module):
    def __init__(self):
        super(Baihua, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)

        )

    def forward(self,x):
        x = self.model1(x)
        return x

baihua = Baihua ()

loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(baihua.parameters(), lr=0.01)#step1

for epoch in range(5):
    running_loss = 0.0
    # one round studying for all data in dataloader
    for data in dataloader:
        imgs, targets = data
        outputs = baihua(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()#set the grad as 0 #step2
        result_loss.backward()#step3, backward propagation to get the grad
        optim.step()#step4, use the grad to update weights
        running_loss += result_loss
    print(running_loss)
    # tensor(18645.9141, grad_fn=<AddBackward0>)
    # tensor(16103.5811, grad_fn=<AddBackward0>)
    # tensor(15426.1211, grad_fn=<AddBackward0>)
    # tensor(16061.1562, grad_fn=<AddBackward0>)
    # tensor(17730.2695, grad_fn=<AddBackward0>)
       
