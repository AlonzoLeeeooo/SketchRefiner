import os
import os.path as osp
import sys
import random
import numpy as np
import cv2

from SRN_src.utils import RandomDeformSketch, generate_stroke_mask

import torch
from torch.utils.data import DataLoader


# train dataset
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, configs):
        super(TrainDataset, self).__init__()

        self.configs = configs

        # deforming algorithm
        self.deform_func = RandomDeformSketch(input_size=configs.size)
        self.max_move = random.randint(configs.max_move_lower_bound, configs.max_move_upper_bound)

        self.image_flist = sorted(self.get_files_from_path(self.configs.images))


    def __len__(self):

        return len(self.image_flist)


    def __getitem__(self, index):

        data = {}
        data['image'] = cv2.imread(self.image_flist[index])
        filename = osp.basename(self.image_flist[index])
        
        if filename.split('.')[1] == "JPEG" or filename.split('.')[1] == "jpg":
            filename = filename.split('.')[0] + '.png'

        prefix = self.image_flist[index].split(filename)[0]

        data['edge'] = cv2.imread(osp.join(self.configs.edges_prefix, filename))
        # data['edge'] = cv2.imread(prefix + filename + '_edge.png')

        # generate free-form mask
        data['mask'] = generate_stroke_mask(im_size=[self.configs.size, self.configs.size])

        # normalize
        # images in range [-1, 1]
        # masks in range [0, 1]
        # edges in range [0, 1]
        data['image'] = data['image'] / 255.
        data['edge'] = data['edge'] / 255.        

        # resize
        data['image'] = cv2.resize(data['image'], (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)
        data['edge'] = cv2.resize(data['edge'], (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)

        # binarize
        thresh = random.uniform(0.65, 0.75)
        _, data['edge'] = cv2.threshold(data['edge'], thresh=thresh, maxval=1.0, type=cv2.THRESH_BINARY)
        
        # to tensor
        # [H, W, C] -> [C, H, W]
        data['image'] = torch.from_numpy(data['image'].astype(np.float32)).permute(2,0,1).contiguous()
        data['mask'] = torch.from_numpy(data['mask'].astype(np.float32)).permute(2,0,1).contiguous()
        data['edge'] = torch.from_numpy(data['edge'].astype(np.float32)).permute(2,0,1).contiguous()

        # generate deform sketches
        data['sketch'] = self.deform_func(data['edge'].unsqueeze(0), self.max_move).squeeze(0)

        # compress RGB channels to 1 ([C, H, W])
        data['sketch'] = torch.sum(data['sketch'] / 3, dim=0, keepdim=True)
        data['edge'] = torch.sum(data['edge'] / 3, dim=0, keepdim=True)

        # return data consisting of: image, mask, sketch, edge
        data['sketch'] = data['sketch'].detach()
        return data


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list


    def get_files_from_path(self, path):

        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret


# validation dataset
class ValDataset(torch.utils.data.Dataset):
    
    def __init__(self, configs):
        super(ValDataset, self).__init__()

        self.configs = configs

        self.deform_func = RandomDeformSketch(input_size=configs.size)
        self.max_move = random.randint(30, 100)

        self.image_flist = sorted(self.get_files_from_txt(self.configs.images_val))
        self.mask_flist = sorted(self.get_files_from_txt(self.configs.masks_val))


    def __len__(self):

        return len(self.image_flist)


    def __getitem__(self, index):

        data = {}
        data['image'] = cv2.imread(self.image_flist[index])

        filename = osp.basename(self.image_flist[index])

        if filename.split('.')[1] == "JPEG":
            filename = filename.split('.')[0] + '.png'

        data['edge'] = cv2.imread(osp.join(self.configs.edges_prefix_val, filename))

        # generate free-form mask
        data['mask'] = cv2.imread(self.mask_flist[index])

        # normalize
        # images in range [-1, 1]
        # masks in range [0, 1]
        # edges in range [0, 1]
        data['image'] = data['image'] / 255.
        data['mask'] = data['mask'] / 255.
        data['edge'] = data['edge'] / 255.        

        # resize
        data['image'] = cv2.resize(data['image'], (self.configs.size, self.configs.size))
        data['mask'] = cv2.resize(data['mask'], (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)
        data['edge'] = cv2.resize(data['edge'], (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)

        # binarize
        thresh = random.uniform(0.65, 0.75)
        _, data['mask'] = cv2.threshold(data['mask'], thresh=0.5, maxval=1.0, type=cv2.THRESH_BINARY)
        _, data['edge'] = cv2.threshold(data['edge'], thresh=thresh, maxval=1.0, type=cv2.THRESH_BINARY)
        
        # to tensor
        # [H, W, C] -> [C, H, W]
        data['image'] = torch.from_numpy(data['image'].astype(np.float32)).permute(2,0,1).contiguous()
        data['mask'] = torch.from_numpy(data['mask'].astype(np.float32)).permute(2,0,1).contiguous()
        data['edge'] = torch.from_numpy(data['edge'].astype(np.float32)).permute(2,0,1).contiguous()

        # compress RGB channels to 1 ([C=1, H, W])
        data['mask'] = torch.sum(data['mask'] / 3, dim=0, keepdim=True)
        data['edge'] = torch.sum(data['edge'] / 3, dim=0, keepdim=True)

        # generate deform sketches
        data['sketch'] = self.deform_func(data['edge'].unsqueeze(0), self.max_move).squeeze(0).detach()

        data['filename'] = filename
        # return data consisting of: image, mask, sketch, edge, filename
        return data


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()


        return file_list


    def get_files(self, path):

        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret


if __name__ == '__main__':
    pass