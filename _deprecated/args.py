import argparse


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='data',
                        help='root path to data directory')
    parser.add_argument('-s', '--source-names', type=str, required=True, nargs='+',
                        help='source datasets (delimited by space)')
    parser.add_argument('-t', '--target-names', type=str, required=True, nargs='+',
                        help='target datasets (delimited by space)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (tips: 4 or 8 times number of gpus)')
    parser.add_argument('--split-id', type=int, default=0,
                        help='split index (note: 0-based)')
    parser.add_argument('--height', type=int, default=256,
                        help='height of an image')
    parser.add_argument('--width', type=int, default=128,
                        help='width of an image')
    parser.add_argument('--train-sampler', type=str, default='RandomSampler',
                        help='sampler for trainloader')
    parser.add_argument('--combineall', action='store_true',
                        help='combine all data in a dataset (train+query+gallery) for training')
    
    # ************************************************************
    # Data augmentation
    # ************************************************************ 
    parser.add_argument('--random-erase', action='store_true',
                        help='use random erasing for data augmentation')
    parser.add_argument('--color-jitter', action='store_true',
                        help='randomly change the brightness, contrast and saturation')
    parser.add_argument('--color-aug', action='store_true',
                        help='randomly alter the intensities of RGB channels')

    # ************************************************************
    # Video datasets
    # ************************************************************
    parser.add_argument('--seq-len', type=int, default=15,
                        help='number of images to sample in a tracklet')
    parser.add_argument('--sample-method', type=str, default='evenly',
                        help='how to sample images from a tracklet')
    parser.add_argument('--pool-tracklet-features', type=str, default='avg', choices=['avg', 'max'],
                        help='how to pool features over a tracklet (for video reid)')
    
    # ************************************************************
    # Dataset-specific setting
    # ************************************************************
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help='use labeled images, if false, use detected images')
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help='use classic split by Li et al. CVPR\'14')
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help='use cuhk03\'s metric for evaluation')

    parser.add_argument('--market1501-500k', action='store_true',
                        help='add 500k distractors to the gallery set for market1501')
    
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimization algorithm (see optimizers.py)')
    parser.add_argument('--lr', default=0.0003, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help='weight decay')   
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum factor for sgd and rmsprop')
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help='sgd\'s dampening for momentum')
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help='whether to enable sgd\'s Nesterov momentum')
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help='rmsprop\'s smoothing constant')
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='exponential decay rate for adam\'s first moment')
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help='exponential decay rate for adam\'s second moment')
    
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=60, type=int,
                        help='maximum epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful when restart)')

    parser.add_argument('--train-batch-size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help='test batch size')
    
    parser.add_argument('--always-fixbase', action='store_true',
                        help='always fix base network and only train specified layers')
    parser.add_argument('--fixbase-epoch', type=int, default=0,
                        help='how many epochs to fix base network (only train randomly initialized classifier)')
    parser.add_argument('--open-layers', type=str, nargs='+', default=['classifier'],
                        help='open specified layers for training while keeping others frozen')

    parser.add_argument('--staged-lr', action='store_true',
                        help='set different lr to different layers')
    parser.add_argument('--new-layers', type=str, nargs='+', default=['classifier'],
                        help='newly added layers with default lr')
    parser.add_argument('--base-lr-mult', type=float, default=0.1,
                        help='learning rate multiplier for base layers')

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                        help='stepsize to decay learning rate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay')

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--label-smooth', action='store_true',
                        help='use label smoothing regularizer in cross entropy loss')

    # ************************************************************
    # Hard triplet loss-specific setting
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='number of instances per identity')
    parser.add_argument('--lambda-xent', type=float, default=1,
                        help='weight to balance cross entropy loss')
    parser.add_argument('--lambda-htri', type=float, default=1,
                        help='weight to balance hard triplet loss')
    
    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')
    parser.add_argument('--eval-freq', type=int, default=-1,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--seed', type=int, default=1,
                        help='manual seed')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    parser.add_argument('--save-dir', type=str, default='log',
                        help='path to save log and model weights')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices (useful when using managed clusters)')
    parser.add_argument('--visualize-ranks', action='store_true',
                        help='visualize ranked results, only available in evaluation mode')

    return parser


def image_dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'combineall': parsed_args.combineall,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,
        'cuhk03_labeled': parsed_args.cuhk03_labeled,
        'cuhk03_classic_split': parsed_args.cuhk03_classic_split,
        'market1501_500k': parsed_args.market1501_500k,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
    }


def video_dataset_kwargs(parsed_args):
    """
    Build kwargs for VideoDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'combineall': parsed_args.combineall,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,
        'seq_len': parsed_args.seq_len,
        'sample_method': parsed_args.sample_method,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
        'staged_lr': parsed_args.staged_lr,
        'new_layers': parsed_args.new_layers,
        'base_lr_mult': parsed_args.base_lr_mult,
    }


def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in lr_schedulers.py from
    the parsed command-line arguments.
    """
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma,
    }