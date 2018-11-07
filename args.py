import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        description='Universal argument parser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('-s', '--source', type=str, required=True, nargs='+',
                        help="source datasets (delimited by space)")
    parser.add_argument('-t', '--target', type=str, required=True, nargs='+',
                        help="target datasets (delimited by space)")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (tips: 4 or 8 times number of gpus)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image")
    parser.add_argument('--split-id', type=int, default=0,
                        help="split index (note: 0-based)")

    # ************************************************************
    # Video datasets
    # ************************************************************
    parser.add_argument('--seq-len', type=int, default=15,
                        help="number of images to sample in a tracklet")
    parser.add_argument('--sample', type=str, default='evenly',
                        help="how to sample images from a tracklet")
    
    # ************************************************************
    # CUHK03-specific setting
    # ************************************************************
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help="use labeled images, if false, use detected images")
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help="use classic split by Li et al. CVPR'14")
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help="use cuhk03's metric for evaluation")
    
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help="optimization algorithm (see optimizers.py)")
    
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=60, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful when restart)")
    parser.add_argument('--train-batch', default=32, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=100, type=int,
                        help="test batch size")
    parser.add_argument('--lr', default=0.0003, type=float,
                        help="initial learning rate")
    parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                        help="stepsize to decay learning rate")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay")
    
    parser.add_argument('--fixbase-epoch', default=0, type=int,
                        help="how many epochs to fix base network (only train randomly initialized classifier)")
    parser.add_argument('--fixbase-lr', default=0.0003, type=float,
                        help="learning rate when the base network is frozen")
    parser.add_argument('--freeze-bn', action='store_true',
                        help="freeze running statistics in BatchNorm layers during training")
    
    parser.add_argument('--label-smooth', action='store_true',
                        help="use label smoothing regularizer in cross entropy loss")

    # ************************************************************
    # Hard triplet loss-specific setting
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")
    parser.add_argument('--htri-only', action='store_true',
                        help="only use hard triplet loss")
    parser.add_argument('--lambda-xent', type=float, default=1,
                        help="weight to balance cross entropy loss")
    parser.add_argument('--lambda-htri', type=float, default=1,
                        help="weight to balance hard triplet loss")
    
    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet50')


    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'],
                        help="how to pool features over a tracklet (for video reid)")
    
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=10,
                        help="print frequency")
    parser.add_argument('--seed', type=int, default=1,
                        help="manual seed")
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--load-weights', type=str, default='',
                        help="load pretrained weights but ignores layers that don't match in size")
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluate only")
    parser.add_argument('--eval-step', type=int, default=-1,
                        help="run evaluation for every N epochs (set to -1 to test only in the end)")
    parser.add_argument('--start-eval', type=int, default=0,
                        help="start to evaluate after a specific epoch")
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('--use-cpu', action='store_true',
                        help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help="use available gpus instead of specified devices (useful when using managed clusters)")
    parser.add_argument('--visualize-ranks', action='store_true',
                        help="visualize ranked results, only available in evaluation mode")

    return parser