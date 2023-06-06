import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from SIN_src.dataset import *
from SIN_src.models.SIN_network import *
from SIN_src.config import Config
from SIN_src.utils import *

def get_files_from_path(path):
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


def visualize(data, keys, path):
        
        filename = data['filename']

        # create sample path if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        data_list = []

        for key in keys:
            item = data[key]
            # [B, C=1, H, W] -> [H, W, C=1]
            if item.size(0) == 1:
                item = torch.cat([item, item, item], dim=0)
            item = item[0, :, :, :,].permute(1, 2, 0)
            item = (item * 255.).cpu().detach().numpy().astype(np.uint8)
            data_list.append(item)

        # concate on `width` dimension
        sample = np.concatenate(data_list, axis=1)

        cv2.imwrite(path + f"/{filename}", sample)


def main_worker(gpu, args):

    # create dirs
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'samples')):
        os.makedirs(os.path.join(args.output, 'samples'))
    if not os.path.exists(os.path.join(args.output, 'inpainted_with_sketch')):
        os.makedirs(os.path.join(args.output, 'inpainted_with_sketch'))
    if not os.path.exists(os.path.join(args.output, 'inpainted_with_refined_sketch')):
        os.makedirs(os.path.join(args.output, 'inpainted_with_refined_sketch'))

    rank = args.node_rank * args.gpus + gpu
    torch.cuda.set_device(gpu)

    device = gpu

    # load config file
    config = Config(args.config_path)
    config.MODE = 1
    config.nodes = args.nodes
    config.gpus = args.gpus
    config.GPU_ids = args.GPU_ids
    config.DDP = args.DDP
    if config.DDP:
        config.world_size = args.world_size
    else:
        config.world_size = 1

    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    deform_func = RandomDeformSketch(input_size=args.size)

    # initialize models
    model = TextureRestorationModule()
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.checkpoint)['generator'])
    print('Model loaded.')
    model.eval()

    str_encoder = PartialSketchEncoder()
    str_encoder = str_encoder.to(device)
    str_encoder.eval()
    str_encoder.load_state_dict(torch.load(args.checkpoint)['str_encoder'])
    print('Structure encoder loaded.')
    str_encoder.eval()

    # initialize data
    image_flist = sorted(get_files_from_path(args.images)) 
    mask_flist = sorted(get_files_from_path(args.masks)) 

    print("\n\nStart evaluating...\n\n")

    with torch.no_grad():
        for i in range(len(image_flist)):
            if i >= args.num_samples:
                print('\n\nTesting done...\n\n')
                sys.exit(0)

            image = cv2.imread(image_flist[i])

            filename = os.path.basename(image_flist[i]).split('.')[0]

            mask = cv2.imread(os.path.join(args.masks, filename + '.png'))
            edge = cv2.imread(os.path.join(args.edges, filename + '.png'))
            sketch = cv2.imread(os.path.join(args.sketches, filename + '.png'))

            # read refined sketch directly from local path
            refined_sketch = cv2.imread(os.path.join(args.refined_sketches, filename + '.png'))
            refined_sketch = cv2.resize(refined_sketch, [args.size, args.size])
            refined_sketch = refined_sketch / 255.
            _, refined_sketch = cv2.threshold(refined_sketch, thresh=random.uniform(0.65, 0.75), maxval=1.0, type=cv2.THRESH_BINARY)
            refined_sketch = torch.from_numpy(refined_sketch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()
            refined_sketch = refined_sketch.to(device)

            # resize
            image = cv2.resize(image, [args.size, args.size])
            mask = cv2.resize(mask, [args.size, args.size])
            edge = cv2.resize(edge, [args.size, args.size])
            sketch = cv2.resize(sketch, [args.size, args.size])

            # normalize to [0, 1]
            image = image / 255.
            mask = mask / 255.
            edge = edge / 255.
            sketch = sketch / 255.

            # binarize
            thresh = random.uniform(0.65, 0.75)
            _, mask = cv2.threshold(mask, thresh=0.5, maxval=1.0, type=cv2.THRESH_BINARY)
            _, edge = cv2.threshold(edge, thresh=thresh, maxval=1.0, type=cv2.THRESH_BINARY)
            _, sketch = cv2.threshold(sketch, thresh=thresh, maxval=1.0, type=cv2.THRESH_BINARY)

            # to tensor
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()
            mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()
            edge = torch.from_numpy(edge.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()
            sketch = torch.from_numpy(sketch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).contiguous()

            # deform sketch
            deformed_sketch = sketch.detach() # [B=1, C=3, H, W]

            # single channel version of elements
            sc_edge = torch.sum(edge / 3, dim=1, keepdim=True)
            sc_sketch = torch.sum(sketch / 3, dim=1, keepdim=True)
            sc_mask = torch.sum(mask / 3, dim=1, keepdim=True)

            # to cuda
            image = image.to(device)
            mask = mask.to(device)
            edge = edge.to(device)
            sc_sketch = sc_sketch.to(device)
            sc_edge = sc_edge.to(device)
            sc_mask = sc_mask.to(device)
            deformed_sketch = deformed_sketch.to(device)

            # forward
            refined_sketch_feature = str_encoder(refined_sketch)
            deformed_sketch_feature = str_encoder(deformed_sketch)

            # forward with refined sketch
            masked_image = image * (1 - mask) 
            out_with_refined_sketch = model(torch.cat([masked_image, sc_mask], dim=1), refined_sketch_feature)
            inpainted_with_refined_sketch = mask * out_with_refined_sketch + (1 - mask) * image

            # forward with deformed sketch
            out_with_deformed_sketch = model(torch.cat([masked_image, sc_mask], dim=1), deformed_sketch_feature)
            inpainted_with_deformed_sketch = mask * out_with_deformed_sketch + (1 - mask) * image

            # save sample of sketch
            inpainted_with_deformed_sketch *= 255.0
            inpainted_with_deformed_sketch = inpainted_with_deformed_sketch.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
            cv2.imwrite(args.output + '/' + 'inpainted_with_sketch' + '/' + filename + '.png', inpainted_with_deformed_sketch[0, :, :, :,])

            # save sample of refined sketch
            inpainted_with_refined_sketch *= 255.0
            inpainted_with_refined_sketch = inpainted_with_refined_sketch.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
            cv2.imwrite(args.output + '/' + 'inpainted_with_refined_sketch' + '/' + filename + '.png', inpainted_with_refined_sketch[0, :, :, :,])

            # make data dict
            data = {
                'filename': filename + '.png',
                'image': image,
                'masked_image': masked_image + deformed_sketch * mask,
                'deformed_sketch': deformed_sketch * mask + (1 - mask) * edge,
                'refined_sketch': refined_sketch,
                'edge': edge,
                'inpainted_with_deformed_sketch': mask * out_with_deformed_sketch + (1 - mask) * image,
                'inpainted_with_refined_sketch': mask * out_with_refined_sketch + (1 - mask) * image,
            }

            visualize(data, ['image', 'masked_image', 'deformed_sketch', 'refined_sketch', 'inpainted_with_deformed_sketch', 'inpainted_with_refined_sketch'], os.path.join(args.output, 'samples'))

            print(f"Progress: {i + 1}/{len(image_flist)}")


    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='path of configuration path')
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    parser.add_argument('--images', type=str, default='', help='path of images for testing')
    parser.add_argument('--masks', type=str, default='', help='path of masks for testing')
    parser.add_argument('--edges', type=str, default='', help='path of edges for testing')
    parser.add_argument('--sketches', type=str, default='', help='path of sketches for testing')
    parser.add_argument('--refined_sketches', type=str, default='', help='path of refined sketches from SRN for testing')
    parser.add_argument('--output', type=str, default='', help='path of output')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained checkpoint path')
    parser.add_argument('--size', type=int, default=256, help='image resolution for testing')
    parser.add_argument('--num_samples', type=int, help='number of samples for testing')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    config_path = args.config_path

    args.config_path = config_path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    if args.DDP:
        args.world_size = args.nodes * args.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22323'
    else:
        args.world_size = 1

    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
