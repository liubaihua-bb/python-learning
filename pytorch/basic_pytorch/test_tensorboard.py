# 2023.4.5
# this period tortured me!!
# the utf-8 code problem： tick the language of the computer (beta utf-8)
# get to know how to use tensorboard
# change the image_path, add_image's "NAME", step'1' and run code -> the tensorboard website will present images after updating.

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "n_dataset/train/bees_image/29494643_e3410f0d37.jpg"
img_PIL = Image.open(image_path)# the type of the image is pil, not suitable for function add_image
img_array = np.array(img_PIL)

# print(img_array.shape) #(512, 768, 3) height, width, channels -> needs to set the dataformats as "HWC"

writer.add_image("train", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()

