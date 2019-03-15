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
    """PRID2011

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
        super(PRID2011, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_dir = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b')

        required_files = [
            self.dataset_dir,
            self.cam_a_dir,
            self.cam_b_dir
        ]
        self.check_before_run(required_files)

        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError('split_id exceeds range, received {}, but expected between 0 and {}'.format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        train = self.process_dir(train_dirs, cam1=True, cam2=True)
        query = self.process_dir(test_dirs, cam1=True, cam2=False)
        gallery = self.process_dir(test_dirs, cam1=False, cam2=True)

        self.init_attributes(train, query, gallery, **kwargs)

        if verbose:
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def process_dir(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets