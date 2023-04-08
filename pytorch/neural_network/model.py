# 2023.4.6
import torch
from torch import nn
# construct the neural network
class Baihua(nn.Module):
    def __init__(self) -> None:
        super(Baihua, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    baihua = Baihua()
    input = torch.ones((64,3,32,32))
    output = baihua(input)
    print(output.shape)