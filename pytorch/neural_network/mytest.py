import torchvision
from torch.utils.data import DataLoader
from model import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter


train_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data[0])#一个tensor图片，一个数字6