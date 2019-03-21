from __future__ import absolute_import
from __future__ import print_function

from .dataset import Dataset, ImageDataset, VideoDataset
from .image import *
from .video import *


__image_datasets = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'prid450s': PRID450S,
    'ilids': iLIDS,
    'sensereid': SenseReID,
    'prid': PRID
}


__video_datasets = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    'dukemtmcvidreid': DukeMTMCVidReID
}


def init_image_dataset(name, **kwargs):
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)


def init_video_dataset(name, **kwargs):
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", but expected to be one of {}'.format(name, avai_datasets))
    return __video_datasets[name](**kwargs)


def add_image_dataset(name, dataset_):
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError('The given name already exists, please choose another name excluding {}'.format(curr_datasets))
    __image_datasets[name] = dataset_


def add_video_dataset(name, dataset_):
    curr_datasets = list(__video_datasets.keys())
    if name in curr_datasets:
        raise ValueError('The given name already exists, please choose another name excluding {}'.format(curr_datasets))
    __video_datasets[name] = dataset_