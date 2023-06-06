import argparse
import random
import os
import cv2
import sys
import numpy as np

import torch

from SRN_src.utils import RandomDeformSketch, binary_value
from SRN_src.SRN_network import RegistrationModule, EnhancementModule

# initialize testing configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Configuration of sketch refinement network')

    parser.add_argument('--images', default='', type=str, help='path of images')
    parser.add_argument('--masks', default='', type=str, help='path prefix of masks')
    parser.add_argument('--edge_prefix', default='', type=str, help='path prefix of edges')
    parser.add_argument('--sketch_prefix', default='', type=str, help='path prefix of sketches')
    parser.add_argument('--size', default=256, type=int, help='image resolution for testing')
    parser.add_argument('--output', default='', type=str, help='path of output')
    parser.add_argument('--num_samples', type=int, help='number of testing images')
    parser.add_argument('--RM_checkpoint', default='', type=str, help='checkpoint path of registration module')
    parser.add_argument('--EM_checkpoint', default='', type=str, help='checkpoint path of enhancement module')

    args = parser.parse_args()

    return args

def get_files_from_path(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# read flist from txt file
def get_files_from_txt(path):
    
    file_list = []
    f = open(path)
    for line in f.readlines():
        line = line.strip("\n")
        file_list.append(line)
        sys.stdout.flush()
    f.close()

    return file_list

def visualize(data, keys, path):
        
        filename = data['filename']

        # create sample path if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        result_path = os.path.join(path, 'results')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        refined_sketch = data['em_out']
        refined_sketch = torch.cat([refined_sketch, refined_sketch, refined_sketch], dim=0)
        refined_sketch = refined_sketch.permute(1, 2, 0)
        refined_sketch *= 255.
        refined_sketch = refined_sketch.cpu().detach().numpy().astype(np.uint8)

        cv2.imwrite(result_path + f"/{filename}", refined_sketch)

        data_list = []

        for key in keys:
            item = data[key]
            # [B, C=1, H, W] -> [H, W, C=1]
            if item.size(0) == 1:
                item = torch.cat([item, item, item], dim=0)
            item = item[:, :, :,].permute(1, 2, 0)
            item = (item * 255.).cpu().detach().numpy().astype(np.uint8)
            data_list.append(item)

        # concate on `width` dimension
        sample = np.concatenate(data_list, axis=1)

        cv2.imwrite(path + f"/{filename}", sample)

if __name__ == '__main__':
    
    configs = parse_args()
    count = 0

    max_move = random.randint(30, 100)
    deform_func = RandomDeformSketch(configs.size)

    # initialize network
    registration_module = RegistrationModule().cuda()
    enhancement_module = EnhancementModule().cuda()

    # load pretrained checkpoint
    registration_module.load_state_dict(torch.load(configs.RM_checkpoint)['parameters'])
    enhancement_module.load_state_dict(torch.load(configs.EM_checkpoint)['parameters'])

    # initialize data
    image_flist = sorted(get_files_from_path(configs.images))
    mask_flist = sorted(get_files_from_path(configs.masks))

    # inference
    for i in range(configs.num_samples):
        image = cv2.imread(image_flist[i])
        
        file_name = os.path.basename(image_flist[i])
        file_name = file_name.split('.')[0] + '.png'
        
        mask = cv2.imread(os.path.join(configs.masks, file_name))

        edge = cv2.imread(os.path.join(configs.edge_prefix, file_name))
        sketch = cv2.imread(os.path.join(configs.sketch_prefix, file_name))

        # resize
        image = cv2.resize(image, (configs.size, configs.size))
        mask = cv2.resize(mask, (configs.size, configs.size))
        sketch = cv2.resize(sketch, (configs.size, configs.size))


        # normalize
        image = image / 255.
        mask = mask / 255.
        edge = edge / 255.
        sketch = sketch / 255.

        # to tensor
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).contiguous()
        edge = torch.from_numpy(edge.astype(np.float32)).permute(2, 0, 1).contiguous()
        sketch = torch.from_numpy(sketch.astype(np.float32)).permute(2, 0, 1).contiguous()

        # cuda
        image = image.cuda()
        mask = mask.cuda()
        edge = edge.cuda()
        sketch = sketch.cuda()

        # compress to single channel
        mask = torch.sum(mask / 3, dim=0, keepdim=True)
        sketch = torch.sum(sketch / 3, dim=0, keepdim=True)
        edge = torch.sum(edge / 3, dim=0, keepdim=True)

        # binarize value
        thresh = random.uniform(0.65, 0.75)
        mask = binary_value(mask, 0.5)
        sketch = binary_value(sketch, thresh)
        edge = binary_value(edge, thresh)

        visualize_sketch = sketch

        # forward
        masked_img = image * (1 - mask) + mask
        sketch = sketch * mask + edge * (1 - mask)

        rm_in = torch.cat([masked_img, mask, sketch], dim=0).unsqueeze(0)
        rm_out = registration_module(rm_in).squeeze(0)

        thresh = torch.mean(rm_out)
        em_in = binary_value(rm_out, thresh)

        em_in = em_in * mask + edge * (1 - mask)
        visualize_em_in = em_in.detach()

        em_out = enhancement_module(em_in.unsqueeze(0)).squeeze(0)
        em_out = torch.clamp(em_out, 0.0, 1.0)
        em_out = em_out * mask + edge * (1 - mask)
        em_out = binary_value(em_out, torch.mean(em_out))

        data = {
            'filename': file_name,
            'image': image,
            'masked_img': masked_img,
            'rm_in': visualize_sketch * mask + (1 - mask) * edge,
            'rm_out': rm_out,
            'edge': edge,
            'em_in': rm_out * mask + (1 - mask) * edge,
            'em_out': em_out * mask + (1 - mask) * edge,
        }

        # visualize
        visualize(data, ['image', 'masked_img', 'rm_in', 'edge', 'em_in', 'em_out'], configs.output)

        count += 1
        print(f"Progress completed: {count}/{configs.num_samples}")


