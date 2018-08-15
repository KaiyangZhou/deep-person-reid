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


class iLIDS(object):
    """
    iLIDS (for single shot setting)

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    URL: http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html
    
    Dataset statistics:
    # identities: 300
    # images: 600
    # cameras: 2
    """
    dataset_dir = 'ilids-vid'

    def __init__(self, root='data', split_id=0, verbose=True, **kwargs):
        super(iLIDS, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(self.dataset_dir, 'i-LIDS-VID')
        self.split_dir = osp.join(self.dataset_dir, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_1_path = osp.join(self.dataset_dir, 'i-LIDS-VID/images/cam1') # differ from video
        self.cam_2_path = osp.join(self.dataset_dir, 'i-LIDS-VID/images/cam2')

        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_imgs, num_train_pids = self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_imgs, num_query_pids = self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_imgs, num_gallery_pids = self._process_data(test_dirs, cam1=False, cam2=True)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs

        if verbose:
            print("=> iLIDS (single-shot) loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
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

        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        urllib.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.dataset_dir)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits ...")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids // 2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = sorted(glob.glob(osp.join(self.cam_1_path, '*')))
            person_cam2_dirs = sorted(glob.glob(osp.join(self.cam_2_path, '*')))

            person_cam1_dirs = [osp.basename(item) for item in person_cam1_dirs]
            person_cam2_dirs = [osp.basename(item) for item in person_cam2_dirs]

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

    def _process_data(self, dirnames, cam1=True, cam2=True):
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        dataset = []
        
        for i, dirname in enumerate(dirnames):
            if cam1:
                pdir = osp.join(self.cam_1_path, dirname)
                img_path = glob.glob(osp.join(pdir, '*.png'))
                # only one image is available in one folder
                assert len(img_path) == 1
                img_path = img_path[0]
                pid = dirname2pid[dirname]
                dataset.append((img_path, pid, 0))

            if cam2:
                pdir = osp.join(self.cam_2_path, dirname)
                img_path = glob.glob(osp.join(pdir, '*.png'))
                # only one image is available in one folder
                assert len(img_path) == 1
                img_path = img_path[0]
                pid = dirname2pid[dirname]
                dataset.append((img_path, pid, 1))

        num_imgs = len(dataset)
        num_pids = len(dirnames)

        return dataset, num_imgs, num_pids
