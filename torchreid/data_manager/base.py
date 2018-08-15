from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import lmdb


class BaseImgDataset(object):
    def __init__(self):
        self.train_lmdb_path = None
        self.query_lmdb_path = None
        self.gallery_lmdb_path = None

    def generate_lmdb(self):
        assert isinstance(self.train, list)
        assert isinstance(self.query, list)
        assert isinstance(self.gallery, list)
        
        print("Reminder: this function is under development, some datasets might not be applicable yet")

        self.train_lmdb_path = osp.join(self.dataset_dir, 'train_lmdb')
        self.query_lmdb_path = osp.join(self.dataset_dir, 'query_lmdb')
        self.gallery_lmdb_path = osp.join(self.dataset_dir, 'gallery_lmdb')

        def _write_lmdb(write_path, data_list):
            if osp.exists(write_path):
                return
            
            print("Generating lmdb files to '{}'".format(write_path))
            
            num_data = len(data_list)
            max_map_size = int(num_data * 500**2 * 3) # be careful with this
            env = lmdb.open(write_path, map_size=max_map_size)
            
            for img_path, pid, camid in data_list:
                with env.begin(write=True) as txn:
                    with open(img_path, 'rb') as imgf:
                        imgb = imgf.read()
                    txn.put(img_path, imgb)

        _write_lmdb(self.train_lmdb_path, self.train)
        _write_lmdb(self.query_lmdb_path, self.query)
        _write_lmdb(self.gallery_lmdb_path, self.gallery)
