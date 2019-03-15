from __future__ import absolute_import
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import copy


class BaseDataset(object):
    """Base class of reid dataset"""
    
    def __init__(self, root):
        self.root = osp.expanduser(root)

    def check_before_run(self, required_files):
        """Check if required files exist before going deeper"""
        for f in required_files:
            if not osp.exists(f):
                raise RuntimeError('"{}" is not found'.format(f))

    def extract_data_info(self, data):
        """Extract info from data list

        Args:
            data (list): contains a list of (img_path, pid, camid)
        """
        raise NotImplementedError

    def get_num_pids(self, data):
        return self.extract_data_info(data)[0]

    def get_num_cams(self, data):
        return self.extract_data_info(data)[2]

    def init_attributes(self, train, query, gallery, combineall=False, **kwargs):
        """Initialize class attributes

        Args:
            train (list): contains a list of (img_path, pid, camid)
            query (list): contains a list of (img_path, pid, camid)
            gallery (list): contains a list of (img_path, pid, camid)
            combineall (bool): if set to True, combine all data for training, default is False

        Notes:
            This method has to be called (at the end) in each dataset class.
        """
        self._train = train
        self._query = query
        self._gallery = gallery
        self._num_train_pids = self.get_num_pids(train)
        self._num_train_cams = self.get_num_cams(train)

        if combineall:
            self._train = self.combine_all(train, query, gallery)
            self._num_train_pids = self.get_num_pids(self.train)

    def combine_all(self, train, query, gallery):
        """Combine all data for training

        Notes:
            1. In general, we assume that all query identities appear in gallery set.
            2. All pids in train have been relabeled (starts from 0)
            3. pid=0 (background) and pid=-1 (junk) are discarded.
            4. Camera views remain the same across train, query and gallery.
        """
        raise NotImplementedError

    @property
    def train(self):
        # train list containing (img_path, pid, camid)
        return self._train

    @property
    def query(self):
        # query list containing (img_path, pid, camid)
        return self._query

    @property
    def gallery(self):
        # gallery list containing (img_path, pid, camid)
        return self._gallery

    @property
    def num_train_pids(self):
        # number of train identities
        return self._num_train_pids

    @property
    def num_train_cams(self):
        # number of train camera views
        return self._num_train_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """Base class of image-reid dataset"""

    def extract_data_info(self, data):
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def combine_all(self, train, query, gallery):
        combined = copy.deepcopy(train)

        # relabel pids in gallery
        g_pids = set()
        for _, pid, _ in gallery:
            if pid==0 or pid==-1:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid==0 or pid==-1:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid))

        _combine_data(query)
        _combine_data(gallery)

        return combined

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.extract_data_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.extract_data_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.extract_data_info(gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, num_train_imgs, num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, num_query_imgs, num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print('  ----------------------------------------')


class BaseVideoDataset(BaseDataset):
    """Base class of video-reid dataset"""

    def extract_data_info(self, data, return_tracklet_stats=False):
        pids = set()
        cams = set()
        tracklet_stats = []
        for img_paths, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
            tracklet_stats += [len(img_paths)]
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.extract_data_info(train, return_tracklet_stats=True)
        
        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.extract_data_info(query, return_tracklet_stats=True)
        
        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.extract_data_info(gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print('  train    | {:5d} | {:11d} | {:9d}'.format(num_train_pids, num_train_tracklets, num_train_cams))
        print('  query    | {:5d} | {:11d} | {:9d}'.format(num_query_pids, num_query_tracklets, num_query_cams))
        print('  gallery  | {:5d} | {:11d} | {:9d}'.format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print('  -------------------------------------------')
        print('  number of images per tracklet: {} ~ {}, average {:.2f}'.format(min_num, max_num, avg_num))
        print('  -------------------------------------------')