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


class SenseReID(object):
    """
    SenseReID

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
        super(SenseReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_probe')
        self.gallery_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_gallery')

        self._check_before_run()

        query, num_query_pids, num_query_imgs, query_pids_set = self._process_dir(self.query_dir)
        gallery, num_gallery_pids, num_gallery_imgs, gallery_pids_set = self._process_dir(self.gallery_dir)

        total_pids = query_pids_set | gallery_pids_set # set union
        num_total_pids = len(total_pids)
        num_total_imgs = num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> SenseReID loaded (test only)")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = copy.deepcopy(query) # only used to initialize trainloader
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_query_pids # only used to initialize reid convnet
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []
        pid_container = set()

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        
        return dataset, num_pids, num_imgs, pid_container