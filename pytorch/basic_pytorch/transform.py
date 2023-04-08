# 2023.4.5
# in this period, i struggled with the conda virtual environment (vscode, jupyter...), finally i gave up the jupter...
# and learn to use the function transforms.ToTensor()
# and be more familiar with the SummaryWriter, namely tensorboard

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# python -> tensor
# 1 how to use transform
# 2 why use tensor type: tensor include the theoretical basic params needed by neural networks

img_path = "n_dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("logs_transform")


# 1 how to use transform
# transforms.py is like a kit, we need to take out the tool
tensor_trans = transforms.ToTensor()# create concrete tool
tensor_img = tensor_trans(img)# and then use the concrete tool

# print(tensor_img)

writer.add_image("tensor_img", tensor_img)

writer.close()

# To launch the tensorboard: tensorboard --logdir=DIR

# import torch
# print(torch.__file__)
#D:\anaconda\anaconda3\lib\site-packages\torch\__init__.py


