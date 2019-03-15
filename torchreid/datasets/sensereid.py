from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
import copy

from .bases import BaseImageDataset


class SenseReID(BaseImageDataset):
    """SenseReID

    This dataset is used for test purpose only.

    Reference:
    Zhao et al. Spindle Net: Person Re-identification with Human Body
    Region Guided Feature Decomposition and Fusion. CVPR 2017.

    URL: https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view

    Dataset statistics:
    - train: 0 ids, 0 images
    - query: 522 ids, 1040 images
    - gallery: 1717 ids, 3388 images
    """
    dataset_dir = 'sensereid'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(SenseReID, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_probe')
        self.gallery_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_gallery')

        required_files = [
            self.dataset_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)
        train = copy.deepcopy(query) # dummy variable

        self.init_attributes(train, query, gallery, **kwargs)

        if verbose:
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            dataset.append((img_path, pid, camid))
        
        return dataset