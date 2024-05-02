from torch.utils.data import Dataset
from PIL import Image

import os
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor
from config import Config


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_image(imfile1, imfile2):
    img1 = Image.open(imfile1).convert('RGB')
    img1 = ToTensor()(img1)
    img2 = Image.open(imfile2).convert('RGB')
    img2 = ToTensor()(img2)
    return img1, img2


class KittiDataset(Dataset):
    def __init__(self, model, main_dir, mode, transform=None):
        self.main_dir = main_dir
        self.mode = mode
        self.transform = transform
        self.model = model

        if self.mode == 'training':
            self.folder_path = os.path.join(self.main_dir, 'training/image_2')
        elif self.mode == 'testing':
            self.folder_path = os.path.join(self.main_dir, 'testing/image_2')
        else:
            raise ValueError("Mode must be 'training' or 'testing'")

        # self.image_files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        self.image_files = []
        for f in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, f)):
                img_name = f.split('_')[0]
                self.image_files.append(img_name)
        self.image_files = list(set(self.image_files))
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img1_path = os.path.join(self.folder_path, img_name + '_10.png')
        img2_path = os.path.join(self.folder_path, img_name + '_11.png')
        image1, image2 = load_image(img1_path, img2_path)
        
        resize = Resize(Config.model_scene_sizes_WH[self.model])
        image1 = resize(image1)
        image2 = resize(image2)
        return image1, image2
