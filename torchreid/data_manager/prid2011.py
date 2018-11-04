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

from torchreid.utils.iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseVideoDataset


class PRID2011(BaseVideoDataset):
    """
    PRID2011

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.

    URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2
    """
    dataset_dir = 'prid2011'

    def __init__(self, root='data', split_id=0, min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_path = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b')

        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train = self._process_data(train_dirs, cam1=True, cam2=True)
        query = self._process_data(test_dirs, cam1=True, cam2=False)
        gallery = self._process_data(test_dirs, cam1=False, cam2=True)

        if verbose:
            print("=> PRID2011 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, _, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, _, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, _, self.num_gallery_cams = self.get_videodata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets