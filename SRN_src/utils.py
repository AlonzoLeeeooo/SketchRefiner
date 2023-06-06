import os
import cv2
import sys
import argparse
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


if __name__ == '__main__':

    # verbose free form masking algorithm
    # return [size, size, 1] numpy.array
    mask = generate_stroke_mask([256, 256])
    print(mask.shape)

    # verbose deforming algorithm
    x = torch.randn(1, 256, 256)
    deform_func = RandomDeformSketch(input_size=256)
    y = deform_func(x.unsqueeze(0), 0).squeeze(0)
    print(y.size())

    print(x)
    z = binary_value(x, 0.5)
    print(z)