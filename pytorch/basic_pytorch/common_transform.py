# 2023.4.5
# learn the common transform function: totensor, normalize, resize, compose, randomcrop
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs_transform")

img = Image.open("n_dataset/train/ants_image/5650366_e22b7e1065.jpg")
# print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("ToTensor",img_tensor, 1)

# Normalize
# print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)#also need tensor type data
# print(img_norm[0][0][0])

writer.add_image("Normalize",img_norm)


# Resize
# print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
# print(img_resize.size)
# trans_resize return a PIL image, so if we want to writer.add, we should transform it to tensor type
img_resize = trans_totensor(img_resize)

writer.add_image("Resize",img_resize)


# Compose - resize - 2
# in the list of the compose, the former function's output should equal to the latter's input
# compose: compose many functions, run them in order
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)

writer.add_image("Resize", img_resize_2, 1)


# RandomCrop
trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()