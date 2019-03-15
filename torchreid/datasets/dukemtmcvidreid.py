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


class DukeMTMCVidReID(BaseVideoDataset):
    """DukeMTMCVidReID

    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.

    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID
    
    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train) + 2636 (test)
    """
    dataset_dir = 'dukemtmc-vidreid'

    def __init__(self, root='data', min_seq_len=0, verbose=True, **kwargs):
        super(DukeMTMCVidReID, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/gallery')
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')
        self.min_seq_len = min_seq_len

        self.download_data()

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query = self.process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery = self.process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        self.init_attributes(train, query, gallery, **kwargs)

        if verbose:
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def download_data(self):
        if osp.exists(self.dataset_dir):
            return

        print('Creating directory {}'.format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print('Downloading DukeMTMC-VideoReID dataset')
        urllib.urlretrieve(self.dataset_url, fpath)

        print('Extracting files')
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets']

        print('=> Generating split json file (** this might take a while **)')
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print('Processing "{}" with {} person identities'.format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print('Warn: index name {} in {} is missing, jump to next'.format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        print('Saving split to {}'.format(json_path))
        split_dict = {
            'tracklets': tracklets,
        }
        write_json(split_dict, json_path)

        return tracklets