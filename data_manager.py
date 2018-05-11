from __future__ import print_function, absolute_import
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

from utils import mkdir_if_missing, write_json, read_json

"""Image ReID"""

class Market1501(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

class CUHK03(object):
    """
    CUHK03

    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!
    
    Dataset statistics:
    # identities: 1360
    # images: 13164
    # cameras: 6
    # splits: 20 (classic)

    Args:
        split_id (int): split index (default: 0)
        cuhk03_labeled (bool): whether to load labeled images; if false, detected images are loaded (default: False)
    """
    dataset_dir = 'cuhk03'

    def __init__(self, root='data', split_id=0, cuhk03_labeled=False, cuhk03_classic_split=False, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')
        
        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')
        
        self.split_classic_det_json_path = osp.join(self.dataset_dir, 'splits_classic_detected.json')
        self.split_classic_lab_json_path = osp.join(self.dataset_dir, 'splits_classic_labeled.json')
        
        self.split_new_det_json_path = osp.join(self.dataset_dir, 'splits_new_detected.json')
        self.split_new_lab_json_path = osp.join(self.dataset_dir, 'splits_new_labeled.json')
        
        self.split_new_det_mat_path = osp.join(self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.split_new_lab_mat_path = osp.join(self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')

        self._check_before_run()
        self._preprocess()

        if cuhk03_labeled:
            image_type = 'labeled'
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            image_type = 'detected'
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(split_id, len(splits))
        split = splits[split_id]
        print("Split index = {}".format(split_id))

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        num_train_pids = split['num_train_pids']
        num_query_pids = split['num_query_pids']
        num_gallery_pids = split['num_gallery_pids']
        num_total_pids = num_train_pids + num_query_pids

        num_train_imgs = split['num_train_imgs']
        num_query_imgs = split['num_query_imgs']
        num_gallery_imgs = split['num_gallery_imgs']
        num_total_imgs = num_train_imgs + num_query_imgs

        print("=> CUHK03 ({}) loaded".format(image_type))
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.raw_mat_path):
            raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
        if not osp.exists(self.split_new_det_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
        if not osp.exists(self.split_new_lab_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))

    def _preprocess(self):
        """
        This function is a bit complex and ugly, what it does is
        1. Extract data from cuhk-03.mat and save as png images.
        2. Create 20 classic splits. (Li et al. CVPR'14)
        3. Create new split. (Zhong et al. CVPR'17)
        """
        print("Note: if root path is changed, the previously generated json files need to be re-generated (delete them first)")
        if osp.exists(self.imgs_labeled_dir) and \
           osp.exists(self.imgs_detected_dir) and \
           osp.exists(self.split_classic_det_json_path) and \
           osp.exists(self.split_classic_lab_json_path) and \
           osp.exists(self.split_new_det_json_path) and \
           osp.exists(self.split_new_lab_json_path):
            return

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print("Extract image data from {} and save as png".format(self.raw_mat_path))
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = [] # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                # skip empty cell
                if img.size == 0 or img.ndim < 3: continue
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(campid+1, pid+1, viewid, imgid+1)
                img_path = osp.join(save_dir, img_name)
                imsave(img_path, img)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(name):
            print("Processing {} images (extract and save) ...".format(name))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if name == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[name][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(camp[pid,:], campid, pid, imgs_dir)
                    assert len(img_paths) > 0, "campid{}-pid{} has no images".format(campid, pid)
                    meta_data.append((campid+1, pid+1, img_paths))
                print("done camera pair {} with {} identities".format(campid+1, num_pids))
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):
                
                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(osp.basename(img_path).split('_')[2])
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(osp.basename(img_path).split('_')[2])
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print("Creating classic splits (# = 20) ...")
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })
        
        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2])
                pid = pids[idx]
                if relabel: pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1 # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(filelist, pids, pid2label, train_idxs, img_dir, relabel=True)
            query_info = _extract_set(filelist, pids, pid2label, query_idxs, img_dir, relabel=False)
            gallery_info = _extract_set(filelist, pids, pid2label, gallery_idxs, img_dir, relabel=False)
            return train_info, query_info, gallery_info

        print("Creating new splits for detected images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path),
            self.imgs_detected_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        write_json(splits, self.split_new_det_json_path)

        print("Creating new splits for labeled images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path),
            self.imgs_labeled_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        write_json(splits, self.split_new_lab_json_path)

class DukeMTMCreID(object):
    """
    DukeMTMC-reID

    Reference:
    1. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    2. Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: https://github.com/layumi/DukeMTMC-reID_evaluation
    
    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """
    dataset_dir = 'dukemtmc-reid'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> DukeMTMC-reID loaded")
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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

class MSMT17(object):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html
    
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V1/train')
        self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V1/test')
        self.list_train_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_gallery.txt')

        self._check_before_run()
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.list_train_path)
        #val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query, num_query_pids, num_query_imgs = self._process_dir(self.test_dir, self.list_query_path)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.test_dir, self.list_gallery_path)

        #train += val
        #num_train_imgs += num_val_imgs

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> MSMT17 loaded")
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)
        num_imgs = len(dataset)
        num_pids = len(pid_container)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset, num_pids, num_imgs

"""Video ReID"""

class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    URL: http://www.liangzheng.com.cn/Project/project_mars.html
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    """
    dataset_dir = 'mars'

    def __init__(self, root='data', min_seq_len=0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_name_path = osp.join(self.dataset_dir, 'info/train_name.txt')
        self.test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.dataset_dir, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.dataset_dir, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')

        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    URL: http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2
    """
    dataset_dir = 'ilids-vid'

    def __init__(self, root='data', split_id=0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(self.dataset_dir, 'i-LIDS-VID')
        self.split_dir = osp.join(self.dataset_dir, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_1_path = osp.join(self.dataset_dir, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.dataset_dir, 'i-LIDS-VID/sequences/cam2')

        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
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

        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

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
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

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

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.

    URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2
    """
    dataset_dir = 'prid2011'

    def __init__(self, root='data', split_id=0, min_seq_len=0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
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

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class DukeMTMCVidReID(object):
    """
    DukeMTMCVidReID

    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.

    URL: https://github.com/Yu-Wu/Exploit-Unknown-Gradually
    
    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train) + 2636 (test)
    """
    dataset_dir = 'dukemtmc-vidreid'

    def __init__(self, root='data', min_seq_len=0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'dukemtmc_videoReID/train_split')
        self.query_dir = osp.join(self.dataset_dir, 'dukemtmc_videoReID/query_split')
        self.gallery_dir = osp.join(self.dataset_dir, 'dukemtmc_videoReID/gallery_split')
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self.min_seq_len = min_seq_len
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
                camid = int(img_name[5]) - 1 # index-0
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

"""Create dataset"""

__img_factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
}

__vid_factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'dukemtmcvidreid': DukeMTMCVidReID,
}

def get_names():
    return __img_factory.keys() + __vid_factory.keys()

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

def init_vid_dataset(name, **kwargs):
    if name not in __vid_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __vid_factory.keys()))
    return __vid_factory[name](**kwargs)
