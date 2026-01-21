
from __future__ import print_function, absolute_import
from __future__ import division, print_function, absolute_import
import os
import json
from collections import defaultdict
import random
from ..dataset import ImageDataset
import os.path as osp
import warnings

import shutil


class FRIDAimg(ImageDataset):
    """
    FRIDA Dataset
    Args:
        data_dir (str): Path to the root directory of FRIDA dataset.
        min_seq_len (int): Tracklet with length shorter than this value will be discarded (default: 0).
    """
    
    data_dir = ''
    data_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.join(self.root, self.data_dir)
        self.train_dirs = [f"Segment_{i + 1}" for i in range(4)]  # FRIDA has 4 segments
        self.cameras = ['Camera_1', 'Camera_2', 'Camera_3']  # FRIDA has 3 cameras
        self._check_before_run()

        train, test, num_train_tracklets, num_test_tracklets, num_train_pids, num_test_pids, selected_persons_test = \
            self._process_data(self.train_dirs, min_seq_len=0, num_train_ids=10)

        self.train = train
        self.test = test
        self.num_train_pids = num_train_pids
        self.num_test_pids = num_test_pids
        self.num_train_cams = len(self.cameras)
        self.num_test_cams = len(self.cameras)
        self.num_train_vids = num_train_tracklets
        self.num_test_vids = num_test_tracklets

        print(f"First 3 tracklets in train IMAGES: {self.train[:3]}")

        query, gallery, tracklet_query, tracklet_gallery, num_query_pids, num_gallery_pids = \
            self._create_query_gallery(self.test, selected_persons_test)
        self.query = query
        self.gallery = gallery
        print(f"First 3 tracklets in query IMAGES: {self.query[:3]}")
        print(f"First 3 tracklets in gallery IMAGES: {self.gallery[:3]}")
        

        num_query_tracklets = tracklet_query
        num_gallery_tracklets = tracklet_gallery

        super(FRIDAimg, self).__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

   
    def _process_data(self, dirnames, min_seq_len=0, num_train_ids=10):
        tracklets_train = []
        tracklets_test = []
        
        pid_container = list(range(1, 21))  

        # Randomly shuffle the list of person IDs
        # random.shuffle(pid_container)
        
        random.seed(15)

        # Select the first num_train_ids for training, and the rest for testing
        selected_persons_train = random.sample(pid_container, num_train_ids) # pid_container[:num_train_ids]
        selected_persons_test =  [pid for pid in pid_container if pid not in selected_persons_train] # pid_container[num_train_ids:]

        print("selected_persons_train: ", selected_persons_train)
        print("selected_persons_test: ", selected_persons_test)

        # Define label_dict for training set
        labelset = selected_persons_train
        label_dict = {label: index for index, label in enumerate(labelset)}

        # Keep track of person IDs already added to the training and testing sets
        added_persons_train = set()
        added_persons_test = set()

        for segment in dirnames:
            for camera in self.cameras:
                json_file = os.path.join(self.data_dir, 'Annotations', segment, camera, 'data2.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)

                for person_info in data:
                    img_id = person_info['image_id']
                    pid = person_info['person_id']
                    person_id = f'person_{str(pid).zfill(2)}'  # Convert integer ID to zero-padded string

                    for camera_idx in range(len(self.cameras)):
                        camera_name = f'Camera_{camera_idx + 1}'
                        img_path = os.path.join(self.data_dir, 'BBs', segment, img_id, camera_name, f'{person_id}.jpg')

                        if os.path.exists(img_path):
                            if pid in selected_persons_train:
                                if len(tracklets_train) < 100000:
                                    if pid not in added_persons_train:
                                        tracklet = (img_path, label_dict[pid], camera_idx)
                                        tracklets_train.append(tracklet)
                                        added_persons_train.add(pid)
                                    elif pid in added_persons_train and len(added_persons_train) == len(selected_persons_train): 
                                        tracklet = (img_path, label_dict[pid], camera_idx)
                                        tracklets_train.append(tracklet)                     
                            elif pid in selected_persons_test:
                                #if len(tracklets_test) < 80000:
                                if pid not in added_persons_test:
                                    tracklet = (img_path, pid, camera_idx)
                                    tracklets_test.append(tracklet)
                                    added_persons_test.add(pid)
                                elif pid in added_persons_test and len(added_persons_test) == len(selected_persons_test): 
                                    tracklet = (img_path, pid, camera_idx)
                                    tracklets_test.append(tracklet)
                                    
                                

        num_train_tracklets = len(tracklets_train)
        num_test_tracklets = len(tracklets_test)
        num_train_pids = len(added_persons_train)
        num_test_pids = len(added_persons_test)
         

        return tracklets_train, tracklets_test, num_train_tracklets, num_test_tracklets, \
            num_train_pids, num_test_pids, selected_persons_test




    def _create_query_gallery(self, tracklets, selected_persons):
        query = []
        gallery = []
        tracklet_query, tracklet_gallery = 0, 0
        num_query_pids, num_gallery_pids = set(), set()
                     
        for tracklet in tracklets:
            img_path, person_id, camera_idx = tracklet
            if person_id == 5 and camera_idx != 0:
                print(" REQUIRED TRACKLET FOUND!")

            if camera_idx == 0:  # Camera A (query)
                if tracklet_query < 15000:
                    query.append(tracklet)
                    tracklet_query += 1
                    num_query_pids.add(person_id)
            else:  # Other cameras (gallery)
                if tracklet_gallery < 30000:
                    gallery.append(tracklet)
                    tracklet_gallery += 1
                    num_gallery_pids.add(person_id)
        
        print("num_query_pids: ", list(num_query_pids))
        print("num_gallery_pids: ", list(num_gallery_pids))
        print("selected persons: ", selected_persons)

        num_query_pids = len(list(num_query_pids))
        num_gallery_pids = len(list(num_gallery_pids))

        return query, gallery, tracklet_query, tracklet_gallery, num_query_pids, num_gallery_pids
        

    


