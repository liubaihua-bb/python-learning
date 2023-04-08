from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir #root_dir = "dataset/train"
        self.label_dir = label_dir #label_dir = "ants"
        self.path = os.path.join(self.root_dir, self.label_dir) #join two dir
        self.img_path = os.listdir(self.path) #get the list of the images under the dir
    
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)# get the image
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# img, label = bees_dataset[1]
# img.show()

train_dataset = ants_dataset + bees_dataset

# print(len(ants_dataset))#124
# print(len(bees_dataset))#121
# print(len(train_dataset))#245
