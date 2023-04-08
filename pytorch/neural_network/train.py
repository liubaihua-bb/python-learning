# 2023.4.6
# 完整的模型训练！！！

# prepare the dataset
import torchvision
from torch.utils.data import DataLoader
from model import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter


train_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"length of the train dataset:{train_data_size}")
print(f"length of the test dataset:{test_data_size}")

train_data_loader = DataLoader(train_data, batch_size=64, drop_last=True)
test_data_loader = DataLoader(test_data, batch_size=64, drop_last=True)

# construct the neural network
baihua = Baihua()

# loss function
loss_fn = nn.CrossEntropyLoss()

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

# Files already downloaded and verified
# Files already downloaded and verified
# length of the train dataset:50000
# length of the test dataset:10000
# -------Epoch: 1------
# training frequency: 100, loss: 2.291593313217163
# training frequency: 200, loss: 2.2909867763519287
# training frequency: 300, loss: 2.2780747413635254
# training frequency: 400, loss: 2.2226078510284424
# training frequency: 500, loss: 2.1416659355163574
# training frequency: 600, loss: 2.0628795623779297
# training frequency: 700, loss: 2.033020257949829
# total loss on test dataset: 308.2217712402344
# -------Epoch: 2------
# training frequency: 800, loss: 1.946020483970642
# training frequency: 900, loss: 2.09158992767334
# training frequency: 1000, loss: 1.9133672714233398
# training frequency: 1100, loss: 1.6546719074249268
# training frequency: 1200, loss: 1.736380696296692
# training frequency: 1300, loss: 1.66599440574646
# training frequency: 1400, loss: 1.7283644676208496
# training frequency: 1500, loss: 1.6595884561538696
# total loss on test dataset: 274.6632385253906
# -------Epoch: 3------
# training frequency: 1600, loss: 1.58800208568573
# training frequency: 1700, loss: 1.783660650253296
# training frequency: 1800, loss: 1.70176100730896
# training frequency: 1900, loss: 1.5242315530776978
# training frequency: 2000, loss: 1.7346241474151611
# training frequency: 2100, loss: 1.6140738725662231
# training frequency: 2200, loss: 1.6872427463531494
# training frequency: 2300, loss: 1.7704131603240967
# total loss on test dataset: 244.1885528564453
# -------Epoch: 4------
# training frequency: 2400, loss: 1.6746740341186523
# training frequency: 2500, loss: 1.609848141670227
# training frequency: 2600, loss: 1.509031891822815
# training frequency: 2700, loss: 1.4838967323303223
# training frequency: 2800, loss: 1.580591082572937
# training frequency: 2900, loss: 1.6291189193725586
# training frequency: 3000, loss: 1.564083218574524
# training frequency: 3100, loss: 1.5208806991577148
# total loss on test dataset: 232.41957092285156
# -------Epoch: 5------
# training frequency: 3200, loss: 1.541562557220459
# training frequency: 3300, loss: 1.5519145727157593
# training frequency: 3400, loss: 1.3772428035736084
# training frequency: 3500, loss: 1.3572825193405151
# training frequency: 3600, loss: 1.6174273490905762
# training frequency: 3700, loss: 1.3799691200256348
# training frequency: 3800, loss: 1.6211477518081665
# training frequency: 3900, loss: 1.2577539682388306
# total loss on test dataset: 220.5128631591797