from __future__ import print_function, absolute_import
from __future__ import division, print_function, absolute_import
import os
import json
from collections import defaultdict
import random
from ..dataset import VideoDataset
import os.path as osp
import warnings

class FRIDA(VideoDataset):
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
        self.segments = [f"Segment_{i + 1}" for i in range(4)]  # FRIDA has 4 segments
        self.cameras = ['Camera_1', 'Camera_2', 'Camera_3']  # FRIDA has 3 cameras
        self._check_before_run()

        train, test, num_train_tracklets, num_test_tracklets, num_train_pids, num_test_pids = self._process_data(num_train_ids=10)
       
        self.train = train
        self.test = test
        self.num_train_pids = num_train_pids
        self.num_test_pids = num_test_pids
        self.num_train_cams = len(self.cameras)
        self.num_test_cams = len(self.cameras)
        self.num_train_vids = num_train_tracklets
        self.num_test_vids = num_test_tracklets

        # self.train = self._create_query_gallery(self.train)
        print(f"First 3 tracklets in train: {self.train[:3]}")

        query, gallery, tracklet_query, tracklet_gallery, num_query_pids, num_gallery_pids =  self._create_query_gallery(self.test)
        self.query = query
        self.gallery = gallery
        print(f"First 3 tracklets in query: {self.query[:3]}")
        
        num_query_tracklets = tracklet_query
        num_gallery_tracklets =  tracklet_gallery
        

        # print("=> FRIDA loaded")
        # print("Dataset statistics:")
        # print("  ------------------------------")
        # print("  subset   | # ids | # tracklets")
        # print("  ------------------------------")
        # print("  train    | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_vids))
        # print("  test     | {:5d} | {:8d}".format(self.num_test_pids, self.num_test_vids))
        # print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        # print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))


        super(FRIDA, self).__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

   
    def _get_imgIDs(self):
        person_imgIDs = {}

        for segment in self.segments:
            for camera in self.cameras:
                json_file = os.path.join(self.data_dir, 'Annotations', segment, camera, 'data2.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)

                for person_info in data:
                    img_id = person_info['image_id']
                    pid = person_info['person_id']

                    if pid not in person_imgIDs:
                        person_imgIDs[pid] = []

                    person_imgIDs[pid].append(img_id)

        return person_imgIDs

    
    def _get_tracklets(self, person_imgIDs):

        tracklets = []

        for pid, img_ids in person_imgIDs.items():
            # Sort img_ids for the current person
            sorted_img_ids = sorted(img_ids)
            person_id = f'person_{str(pid).zfill(2)}'  # Convert integer ID to zero-padded string

            # Create tracklets for each subsequent set of 3 img_ids
            for i in range(0, len(sorted_img_ids) - 2, 3):
                for segment in self.segments:
                    for camera_idx in range(1, 4):
                        camera_name = f'Camera_{camera_idx}'
                        tracklet_images = []
                        
                        for j in range(3):
                            img_id = sorted_img_ids[i + j]
                            img_path = os.path.join(self.data_dir, 'BBs', segment, img_id, camera_name, f'{person_id}.jpg')
                            
                            if os.path.exists(img_path) and img_path not in tracklet_images:
                                tracklet_images.append(img_path)

                        # Check if the tracklet has at least 2 images before adding it
                        if len(tracklet_images) >= 2:
                            # Create a tracklet as ((img1_path, img2_path, img3_path), person_id, camera_index)
                            tracklet = (tuple(tracklet_images), pid, camera_idx)
                            tracklets.append(tracklet)

        return tracklets


    
    def _process_data(self, num_train_ids=10):
        tracklets_train = []
        tracklets_test = []
        pid_container = list(range(1, 21))  # Assuming 20 persons in total

        random.seed(42)

        # Select 10 random IDs for training
        selected_persons_train = random.sample(pid_container, num_train_ids)

        # Select 10 different random IDs for testing
        selected_persons_test = [pid for pid in pid_container if pid not in selected_persons_train]

        print("selected_persons_train: ", selected_persons_train)
        print("selected_persons_test: ", selected_persons_test)

        person_imgIDs = self._get_imgIDs()
        tracklets = self._get_tracklets(person_imgIDs)
        
        # Create a dictionary to map original person IDs to new labels for training
        label_dict = {label: index for index, label in enumerate(selected_persons_train)}
        print(f"pid mapping {label_dict}")

        for tracklet in tracklets: 
            img_paths, pid, camera_idx = tracklet   
            if pid in selected_persons_train:
                mapped_pid = label_dict[pid]
                tracklet = (img_paths, mapped_pid, camera_idx)
                # tracklet = (img_paths, pid, camera_idx)
                tracklets_train.append(tracklet)
                
            elif pid in selected_persons_test:
                tracklets_test.append(tracklet)
                
                
        num_train_tracklets = len(tracklets_train)
        num_test_tracklets = len(tracklets_test)
        num_train_pids = len(selected_persons_train)
        num_test_pids = len(selected_persons_test)

        return tracklets_train, tracklets_test, num_train_tracklets, \
            num_test_tracklets, num_train_pids, num_test_pids


    def _create_query_gallery(self, tracklets):
        
        query = []
        gallery = []

        tracklet_query, tracklet_gallery = 0, 0    
        num_query_pids, num_gallery_pids = set(), set()

        for tracklet in tracklets:
            img_paths, person_id, camera_idx = tracklet
            if camera_idx == 1:  # Camera A
                query.append(tracklet)
                tracklet_query += 1
                num_query_pids.add(person_id)
            else:  # Cameras B and C
                gallery.append(tracklet)
                tracklet_gallery += 1
                num_gallery_pids.add(person_id)

        num_query_pids =  len(list(num_query_pids))
        num_gallery_pids = len(list(num_gallery_pids))

        return query, gallery, tracklet_query, tracklet_gallery, num_query_pids, num_gallery_pids
