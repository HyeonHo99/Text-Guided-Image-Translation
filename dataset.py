from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class CycleGANDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=A_img, image0=B_img)
            A_img = augmentations["image"]
            B_img = augmentations["image0"]

        return A_img, B_img

class CycleGANDataset_single(Dataset):
    def __init__(self, root, transform = None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)
        self.length_dataset = len(self.images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img = self.images[index % self.length_dataset]
        path = os.path.join(self.root,img)
        img = np.array(Image.open(path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img