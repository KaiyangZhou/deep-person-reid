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


class DukeMTMCVidReID(object):
    """
    DukeMTMCVidReID

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
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/gallery')
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self.min_seq_len = min_seq_len
        self._download_data()
        self._check_before_run()
        print("Note: if root path is changed, the previously generated json files need to be re-generated (so delete them first)")

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-VideoReID dataset")
        urllib.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
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

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
