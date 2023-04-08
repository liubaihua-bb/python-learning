# 2023.4.6
# 完整的模型验证套路
# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# 4 deer
# 5 dog
# 6 frog
# 7 horse
# 8 ship
# 9 truck

import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./imgs/005.png"
image = Image.open(image_path)
# print(image)

# png has 4 channels, so we need to convert it!!!!
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
# print(image.shape)

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

model = torch.load("baihua_29_gpu.pth", map_location=torch.device('cpu'))
# print(model)
image = torch.reshape(image, (1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)
# print(output.argmax(1))
pred = output.argmax(1)
if pred == 0:
    print("airplane")
elif pred == 1:
    print("automobile")
elif pred == 2:
    print("bird")
elif pred == 3:
    print("cat")
elif pred == 4:
    print("deer")
elif pred == 5:
    print("dog")
elif pred == 6:
    print("frog")
elif pred == 7:
    print("horse")
elif pred == 1:
    print("automobile")
elif pred == 8:
    print("ship")
elif pred == 9:
    print("truck")
else:
    print("Wrong!")
