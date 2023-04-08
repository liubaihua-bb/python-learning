# 2023.4.6
# 完整的模型训练！！！
# cuda : 网络模型、数据（输入、标注）、损失函数 .cuda()

# prepare the dataset
import torch
import torchvision
from torch.utils.data import DataLoader
# from model import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# define training devices
# device = torch.device("cpu")
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")


train_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"length of the train dataset:{train_data_size}")
print(f"length of the test dataset:{test_data_size}")

train_data_loader = DataLoader(train_data, batch_size=64, drop_last=True)
test_data_loader = DataLoader(test_data, batch_size=64, drop_last=True)

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
    
baihua = Baihua()
# if torch.cuda.is_available:
#     baihua = baihua.cuda()
baihua = baihua.to(device)


# loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available:
    loss_fn = loss_fn.cuda()
# optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(baihua.parameters(), lr=learning_rate)

# set the parameters of the model
total_train_step = 0 # train frequency
total_test_step = 0 # test frequency

epoch = 3

# tensorboard
writer = SummaryWriter("log_train")

for i in range(epoch):
    print(f"-------Epoch: {i+1}------")

    # TRAIN
    for data in train_data_loader:
        imgs, targets = data
        # if torch.cuda.is_available:
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = baihua(imgs)
        loss = loss_fn(outputs, targets)

        # optimizing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"training frequency: {total_train_step}, loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # TEST
    total_test_loss = 0
    total_accuracy = 0 #整体正确个数
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            # if torch.cuda.is_available:
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = baihua(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()#先对outputs中选logits大的那个的位置【argmax】，然后与targets比较【==】，比较成功的个数加起来【sum】
            total_accuracy += accuracy
    
    print(f"total loss on test dataset: {total_test_loss}")
    print(f"total accuracy on test dataset: {total_accuracy/test_data_size}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

writer.close()
