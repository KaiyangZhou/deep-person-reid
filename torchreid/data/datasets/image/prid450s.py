from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import numpy as np

from torchreid.data.datasets import ImageDataset
from torchreid.utils import read_json, write_json


class PRID450S(ImageDataset):
    """PRID450S

    Reference:
    Roth et al. Mahalanobis Distance Learning for Person Re-Identification. PR 2014.

    URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/prid450s/
    
    Dataset statistics:
        identities: 450
        images: 900
        cameras: 2
    """
    dataset_dir = 'prid450s'
    dataset_url = 'https://files.icg.tugraz.at/f/8c709245bb/?raw=1'

    def __init__(self, root='', split_id=0, min_seq_len=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_a_dir = osp.join(self.dataset_dir, 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'cam_b')
        
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

        super(PRID450S, self).__init__(train, query, gallery, **kwargs)

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