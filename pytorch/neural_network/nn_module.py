# 2023.4.6
import torch
from torch import nn

class Baihua(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(seelf, input):
        output = input + 1
        return output

baihua = Baihua()
x = torch.tensor(1.0)
output = baihua(x)

print(output)