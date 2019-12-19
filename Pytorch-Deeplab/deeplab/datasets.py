import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
import random

class XiangyaTrain(data.Dataset):
    def __init__(self, list_path, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=True, color_jitter=True, rotate=True):
        self.jitter_transform=transforms.Compose([
            transforms.ColorJitter(brightness=64. / 255, contrast=0.25, saturation=0.25, hue=0.04)
        ])
        self.list_path = list_path
        self.ignore_label = 0
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        self.is_jitter = color_jitter
        self.is_rotate = rotate
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file, label_file = name.split(' ')
            img_name = img_file.split('/')[-1]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name,
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 31) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #BGR
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.scale:
            image, label = self.generate_scale_label(image, label)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        if self.is_jitter:
            image = self.jitter_transform(image)
        if self.is_mirror:
            if random.random()<0.5:
                image = F.hflip(image)
                label = F.hflip(label)
        if self.is_rotate:
            angle = random.randint(0, 3)*90
            image = F.rotate(image, angle)
            label = F.rotate(label, angle)
        label = np.asarray(label)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))
        label = np.asarray(label / 127.0, np.uint8)
        return image.copy(), label.copy(), datafiles["name"]


class XiangyaTest(data.Dataset):
    def __init__(self, list_path, crop_size=(321, 321), mean=(128, 128, 128), random_crop=False):
        self.list_path = list_path
        self.random_crop = random_crop
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for img_file in self.img_ids:
            img_name = img_file.split('/')[-1]
            self.files.append({
                "img": img_file,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #BGR
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = image.shape[0:2]

        if self.random_crop and (img_h, img_w)!=(self.crop_h, self.crop_w):
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = np.asarray(image[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        # assert (img_h, img_w) == (self.crop_h, self.crop_w)
        image = image.transpose((2, 0, 1))
        return image.copy(), datafiles["name"]


if __name__ == '__main__':
    pass
