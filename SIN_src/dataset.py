import glob
import os
import sys
import pickle
import random

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import DataLoader


# function to return a path list from a txt file
def get_files_from_txt(path):
    file_list = []
    f = open(path)
    for line in f.readlines():
        line = line.strip("\n")
        file_list.append(line)
        sys.stdout.flush()
    f.close()

    return file_list

def to_int(x):
    return tuple(map(int, x))


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    
    return mask


# generate free form mask
def generate_stroke_mask(im_size, parts=4, maxVertex=25, maxLength=80, maxBrushWidth=40, maxAngle=360):
    
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)

    return mask
    
# make tensor binary value
def binary_value(tensor, thresh):
    indice = (tensor > thresh)
    tensor[indice] = 1.0
    tensor[~indice] = 0.0

    return tensor


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, flist, sketch_path, batch_size,
                 input_size=None, default_size=256, str_size=256,
                 world_size=1,
                 round=1):
        super(TrainDataset, self).__init__()
        self.sketch_path = sketch_path

        self.batch_size = batch_size
        self.round = round  # for places2 round is 32

        self.data = []
        self.sketches = []
        
        f = open(flist, 'r')
        for i in f.readlines():
            i = i.strip()
            self.data.append(i)
        f.close()

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size
        self.str_size = str_size
        self.world_size = world_size

        self.ones_filter = np.ones((3, 3), dtype=np.float32)
        self.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        self.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        if self.training:
            barrel_num = int(len(self.data) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(32, 65,
                                                    step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
                                     256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)
        else:
            self.input_size = self.default_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # resize/crop if needed
        img = self.resize(img, size, size)

        # load filename
        name = os.path.basename(self.data[index])
        name = name.split('.')[0] + '.png'
        prefix = self.data[index].split(name)[0]

        # load mask
        mask = generate_stroke_mask([size, size])
        _, mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY) 

        # load sketch
        sketch = cv2.imread(os.path.join(self.sketch_path, name))
        # sketch = cv2.imread(prefix + name + '_edge.png')
        sketch = cv2.resize(sketch, [size, size])
        _, sketch = cv2.threshold(sketch, thresh=127.5, maxval=255., type=cv2.THRESH_BINARY)


        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['mask'] = self.to_tensor(mask)
        batch['sketch'] = self.to_tensor(sketch)

        batch['name'] = name # + '.png'

        return batch


    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, flist, sketch_path, mask_path, batch_size,
                 input_size=None, default_size=256, str_size=256,
                 world_size=1,
                 round=1):
        super(ValidationDataset, self).__init__()
        self.mask_path = mask_path
        self.sketch_path = sketch_path

        self.batch_size = batch_size
        self.round = round  # for places2 round is 32

        self.data = []
        self.masks = []
        self.sketches = []
        
        f = open(flist, 'r')
        for i in f.readlines():
            i = i.strip()
            self.data.append(i)
        f.close()

        f = open(mask_path, 'r')
        for i in f.readlines():
            i = i.strip()
            self.masks.append(i)
        f.close()

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size
        self.str_size = str_size
        self.world_size = world_size

        self.ones_filter = np.ones((3, 3), dtype=np.float32)
        self.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        self.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        if self.training:
            barrel_num = int(len(self.data) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(32, 65,
                                                    step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
                                     256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)
        else:
            self.input_size = self.default_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # resize/crop if needed
        img = self.resize(img, size, size)

        # load filename
        name = os.path.basename(self.data[index])
        if ".JPEG" in name or ".jpg" in name:
            name = name.split('.')[0] + '.png'

        # load mask
        mask = cv2.imread(self.masks[index])
        mask = cv2.resize(mask, [size, size])
        _, mask = cv2.threshold(mask, 127.5, 255., cv2.THRESH_BINARY) 

        # load sketch
        sketch = cv2.imread(os.path.join(self.sketch_path, name))
        _, sketch = cv2.threshold(sketch, thresh=127.5, maxval=255., type=cv2.THRESH_BINARY)


        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['mask'] = self.to_tensor(mask)
        batch['sketch'] = self.to_tensor(sketch)

        batch['mask'] = torch.sum(batch['mask'] / 3, dim=0, keepdim=True)

        batch['name'] = name

        return batch


    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class LaMaDataset(torch.utils.data.Dataset):
    def __init__(self, flist, batch_size,
                 input_size=None, default_size=256, str_size=256,
                 world_size=1,
                 round=1):
        super(LaMaDataset, self).__init__()

        self.batch_size = batch_size
        self.round = round  # for places2 round is 32

        self.data = []
        self.sketches = []
        
        f = open(flist, 'r')
        for i in f.readlines():
            i = i.strip()
            self.data.append(i)
        f.close()

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size
        self.str_size = str_size
        self.world_size = world_size

        self.ones_filter = np.ones((3, 3), dtype=np.float32)
        self.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        self.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        if self.training:
            barrel_num = int(len(self.data) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(32, 65,
                                                    step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
                                     256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)
        else:
            self.input_size = self.default_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # resize/crop if needed
        img = self.resize(img, size, size)

        # load filename
        name = os.path.basename(self.data[index])
        name = name.split('.')[0]
        prefix = self.data[index].split(name)[0]

        # load mask
        mask = generate_stroke_mask([size, size])
        _, mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY) 

        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['mask'] = self.to_tensor(mask)

        batch['name'] = name + '.png'

        return batch


    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item