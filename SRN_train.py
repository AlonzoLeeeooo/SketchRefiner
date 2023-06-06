import argparse


# initialize configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Configuration of sketch refiner network')

    # data configuration
    ## training data
    parser.add_argument('--images', type=str, default='', help='path of input sketches')
    parser.add_argument('--edges_prefix', type=str, default='', help='path prefix of input edges')
    parser.add_argument('--output', type=str, default='', help='path of output')
    parser.add_argument('--max_move_lower_bound', type=int, default=30, help='lower bound of the randomize interval of deforming algorithm')
    parser.add_argument('--max_move_upper_bound', type=int, default=100, help='upper bound of the randomize interval of deforming algorithm')

    ## validation data
    parser.add_argument('--images_val', type=str, default='', help='path of input sketches for validation')
    parser.add_argument('--masks_val', type=str, default='', help='path of free-form masks for validation')
    parser.add_argument('--sketches_prefix_val', type=str, default='', help='path prefix of deformed sketches for validation')
    parser.add_argument('--edges_prefix_val', type=str, default='', help='path prefix of edges for validation')

    # training configuration
    ## loss function configuration
    parser.add_argument('--rm_l1_weight', default=1.0, type=float, help='the weight of l1 loss of RM')
    parser.add_argument('--rm_cc_weight', default=0.4, type=float, help='the weight of cc loss of RM')
    parser.add_argument('--em_l1_weight', default=1.0, type=float, help='the weight of l1 loss of EM')
    parser.add_argument('--em_cc_weight', default=0.9, type=float, help='the weight of l1 loss of EM')

    ## network configuration
    parser.add_argument('--train_EM', action='store_true', help='train enhancement module, otherwise train registration module')
    parser.add_argument('--RM_checkpoint', default='', type=str, help='checkpoint path of fixed RM')

    ## training configuration
    parser.add_argument('--max_iters', default=500003, type=int, help='max iterations of training')
    parser.add_argument('--epochs', default=10, type=int, help='epochs of training')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number of data loader')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--val_interval', default=0, type=int, help='the interval of validation, set to 0 for no validation')
    parser.add_argument('--sample_interval', default=10000, type=int, help='the interval of saving training samples')
    parser.add_argument('--checkpoint_interval', default=50000, type=int, help='the interval of saving checkpoints')
    parser.add_argument('--size', default=256, type=int, help='resolution of sketches and edges')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    from SRN_src.SRN_trainer import *

    configs = parse_args()

    model = SRNTrainer(configs)
    
    if configs.train_EM:
        model.train_EM()
    else:
        model.train_RM()
