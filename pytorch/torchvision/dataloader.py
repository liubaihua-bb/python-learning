# 2023.4.5

import torchvision 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# the test dataset
test_data = torchvision.datasets.CIFAR10("./cifar10_dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)#drop_last: drop the last pictures whose count does not equal to the batch-size

# the first pcture of the test dataset
# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("log")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"Epoch: {epoch}", imgs, step)# can add a lot of images
        step = step + 1

writer.close()