import copy
import os
import random
import logging

from collections import defaultdict
from ..dataset import ImageDataset

from torchreid.utils import read_json
from torchreid.utils.tools import read_jsonl, write_json

LOGGER = logging.getLogger(__name__)


class MiiMSimulation(ImageDataset):
    # All you need to do here is to generate three lists,
    # which are train, query and gallery.
    # Each list contains tuples of (img_path, pid, camid),
    # where
    # - img_path (str): absolute path to an image.
    # - pid (int): person ID, e.g. 0, 1.
    # - camid (int): camera ID, e.g. 0, 1.
    # Note that
    # - pid and camid should be 0-based.
    # - query and gallery should share the same pid scope (e.g.
    #   pid=0 in query refers to the same person as pid=0 in gallery).
    # - train, query and gallery share the same camid scope (e.g.
    #   camid=0 in train refers to the same camera as camid=0
    #   in query/gallery).

    dataset_dir = "miim_simulation"
    dataset_url = None
    # https://kaiyangzhou.github.io/deep-person-reid/user_guide#use-your-own-dataset

    _num_splits = 10
    _samples_per_obj = 20 # todo make this configurable

    def __init__(self, root="", split_id=0, **kwargs):
        self.manifest_file_path = os.path.join(root, "representation_dataset.jsonl")
        self.guid_file_path = os.path.join(root, "guids.json")
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.root, "train")
        self.query_dir = os.path.join(self.root, "query")
        self.gallery_dir = os.path.join(self.root, "gallery")

        self.split_path = os.path.join(self.dataset_dir, "splits.json")

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        if not os.path.exists(self.query_dir):
            os.makedirs(self.query_dir)

        if not os.path.exists(self.gallery_dir):
            os.makedirs(self.gallery_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        guids = read_json(self.guid_file_path)
        self.prepare_split(self.manifest_file_path, guids)
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but "
                "expected between 0 and {}".format(split_id,
                                                   len(splits) - 1)
            )
        split = splits[split_id]

        train, query, gallery = self.process_split(split)

        super(MiiMSimulation, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self, manifest_fp, pids):
        if not os.path.exists(self.split_path):
            LOGGER.info(self.split_path)
            LOGGER.info("Creating splits...")
            num_pids = len(pids)
            num_train_pids = int(num_pids * 0.5)

            object_manifest_entries_mapping = defaultdict(list)
            for manifest_entry in read_jsonl(manifest_fp):
                path = manifest_entry["path"]
                camera_guid = manifest_entry["camera_guid"]
                object_guid = manifest_entry["object_guid"]

                object_manifest_entries_mapping[manifest_entry["object_guid"]].append((path, object_guid, camera_guid))

            splits = []
            for split_ix in range(self._num_splits):
                # randomly choose num_train_pids train IDs and the rest for test IDs
                pids_copy = copy.deepcopy(pids)
                pids_copy = list(pids_copy.values())
                random.shuffle(pids_copy)
                train_pids = pids_copy[:num_train_pids]
                test_pids = pids_copy[num_train_pids:]
                train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

                train = []
                query = []
                gallery = []

                # for train IDs, all images are used in the train set.
                for pid in train_pids:
                    entries = object_manifest_entries_mapping[pid]
                    random.shuffle(entries)
                    for entry in entries:
                        path, object_guid, camera_guid = entry[0], entry[1], entry[2]

                        guid_label = train_pid2label[object_guid]
                        entry = (path, guid_label, camera_guid)
                        train.append(entry)

                # for each test ID, randomly choose images, one for
                # query and the other one for gallery.
                for pid in test_pids:
                    entries = object_manifest_entries_mapping[pid]
                    samples = random.sample(entries, min(self._samples_per_obj, len(entries)))

                    query_sample = samples[0]
                    gallery_samples = samples[1:]
                    query.append(query_sample)
                    gallery.extend(gallery_samples)

                split = {"train": train, "query": query, "gallery": gallery}
                splits.append(split)

            LOGGER.info(f"Totally {len(splits)} splits are created")
            write_json(splits, self.split_path)
            LOGGER.info(f"Split file is saved to {self.split_path}")

    def process_split(self, split):
        return split["train"], split["query"], split["gallery"]
