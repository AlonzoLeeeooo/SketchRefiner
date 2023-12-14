import os
import cv2
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge', default='', type=str)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--grid_size', default=10, type=int)
    parser.add_argument('--fiter_size', default=5, type=int)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--num_samples', default=10, type=int)
    parser.add_argument('--max_move', default=20, type=int)

    args = parser.parse_args()

    return args

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

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

# random deformation    
def make_Gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    temp = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
    return temp/sum(sum(temp))

class RandomDeformSketch(nn.Module):
    def __init__(self, input_size, grid_size=10, fiter_size=5, gpu=True):
        super(RandomDeformSketch, self).__init__()

        self.max_move = random.randint(0, 100)
        self.grid_size = grid_size
        self.fiter_size = fiter_size
        self.input_size = input_size
        self.pad = nn.ReplicationPad2d(self.fiter_size)
        self.gpu = gpu
        
        gaussian_weights = torch.FloatTensor(make_Gaussian(2*self.fiter_size+1, fwhm = self.fiter_size))
        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.fiter_size+1,2*self.fiter_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.input_size, self.input_size)
        for k in range(2):
            for i in range(self.input_size):
                for j in range(self.input_size):
                    self.P_basis[k,i,j] = k*i/(self.input_size-1.0)+(1.0-k)*j/(self.input_size-1.0)
        
    def create_grid(self, x, max_move):
        max_offset = 2.0*max_move/self.input_size
        P = torch.autograd.Variable(torch.zeros(1,2,self.input_size, self.input_size),requires_grad=False)
        # P = P.cuda() if self.gpu else P
        P[0,:,:,:] = self.P_basis*2-1
        P = P.expand(x.size(0),2,self.input_size, self.input_size)
        offset_x = torch.autograd.Variable(torch.randn(x.size(0),1,self.grid_size, 
                                                       self.grid_size))
        offset_y = torch.autograd.Variable(torch.randn(x.size(0),1,self.grid_size, 
                                                       self.grid_size))
        # offset_x = offset_x.cuda() if self.gpu else offset_x
        # offset_y = offset_y.cuda() if self.gpu else offset_y
        offset_x_filter = self.filter(self.pad(offset_x)) * max_offset
        offset_y_filter = self.filter(self.pad(offset_y)) * max_offset
        offset_x_filter = torch.clamp(offset_x_filter,min=-max_offset,max=max_offset)
        offset_y_filter = torch.clamp(offset_y_filter,min=-max_offset,max=max_offset)

        grid = torch.cat((offset_x_filter,offset_y_filter), 1)
        grid = F.interpolate(grid, [self.input_size,self.input_size], mode='bilinear')
        
        grid = torch.clamp(grid + P, min=-1, max=1)
        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)
        return grid

    def forward(self, x, max_move):
        grid = self.create_grid(x, max_move)
        y = F.grid_sample(x, grid)

        return y

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    edge_flist = get_files(args.edge)
    deform_func = RandomDeformSketch(input_size=args.input_size, grid_size=args.grid_size, fiter_size=args.fiter_size, gpu=args.gpu)
    count = 0

    with torch.no_grad():
        for i in edge_flist:
            count += 1
            if count > args.num_samples:
                sys.exit(0)
            print(f"Processing: {count}/{args.num_samples}")
            max_move = random.randint(0, 100)

            x = cv2.imread(i)
            filename = os.path.basename(i)
            x = torch.from_numpy(x.astype(np.float32) / 255.).permute(2, 0, 1)
            x = x.unsqueeze(0)
            
            y = deform_func(x, args.max_move)

            y = y.squeeze(0)
            y = (y * 255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, filename), y)

