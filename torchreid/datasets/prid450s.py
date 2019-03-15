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
from .bases import BaseImageDataset


class PRID450S(BaseImageDataset):
    """PRID450S

    Reference:
    Roth et al. Mahalanobis Distance Learning for Person Re-Identification. PR 2014.

    URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/prid450s/
    
    Dataset statistics:
    # identities: 450
    # images: 900
    # cameras: 2
    """
    dataset_dir = 'prid450s'

    def __init__(self, root='data', split_id=0, min_seq_len=0, verbose=True, **kwargs):
        super(PRID450S, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.dataset_url = 'https://files.icg.tugraz.at/f/8c709245bb/?raw=1'
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_a_dir = osp.join(self.dataset_dir, 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'cam_b')

        self.download_data()
        
        required_files = [
            self.dataset_dir,
            self.cam_a_dir,
            self.cam_b_dir
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError('split_id exceeds range, received {}, but expected between 0 and {}'.format(split_id, len(splits)-1))
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        self.init_attributes(train, query, gallery, **kwargs)

        if verbose:
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def download_data(self):
        if osp.exists(self.dataset_dir):
            return

        print('Creating directory {}'.format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, 'prid_450s.zip')

        print('Downloading PRID450S dataset')
        urllib.urlretrieve(self.dataset_url, fpath)

        print('Extracting files')
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def prepare_split(self):
        if not osp.exists(self.split_path):
            cam_a_imgs = sorted(glob.glob(osp.join(self.cam_a_dir, 'img_*.png')))
            cam_b_imgs = sorted(glob.glob(osp.join(self.cam_b_dir, 'img_*.png')))
            assert len(cam_a_imgs) == len(cam_b_imgs)

            num_pids = len(cam_a_imgs)
            num_train_pids = num_pids // 2

            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = np.sort(order[:num_train_pids])
                idx2label = {idx: label for label, idx in enumerate(train_idxs)}

                train, test = [], []

                # processing camera a
                for img_path in cam_a_imgs:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[1].split('.')[0])
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], 0))
                    else:
                        test.append((img_path, img_idx, 0))

                # processing camera b
                for img_path in cam_b_imgs:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[1].split('.')[0])
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], 1))
                    else:
                        test.append((img_path, img_idx, 1))

                split = {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))