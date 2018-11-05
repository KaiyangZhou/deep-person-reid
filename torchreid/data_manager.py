from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset
from .datasets import init_imgreid_dataset, init_vidreid_dataset


class ImageDataManager(object):

    def __init__(self, train_names, test_names, root, split_id,
                 transform_train, transform_test, train_batch, test_batch,
                 workers, pin_memory, **kwargs):

        self.train_names = train_names
        self.test_names = test_names
        
        self.train = []
        self.num_train_pids = 0
        self.num_train_cams = 0

        print("=> Initializing TRAIN datasets")

        for name in self.train_names:
            dataset = init_imgreid_dataset(root=root, name=name, split_id=split_id, **kwargs)

            for img_path, pid, camid in dataset.train:
                pid += self.num_train_pids
                camid += self.num_train_cams
                self.train.append((img_path, pid, camid))

            self.num_train_pids += dataset.num_train_pids
            self.num_train_cams += dataset.num_train_cams

        self.trainloader = DataLoader(
            ImageDataset(self.train, transform=transform_train),
            batch_size=train_batch, shuffle=True, num_workers=workers,
            pin_memory=pin_memory, drop_last=True
        )

        print("=> Initializing TEST datasets")

        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.test_names}
        for name in self.test_names:
            dataset = init_imgreid_dataset(root=root, name=name, split_id=split_id, **kwargs)

            self.testloader_dict[name]['query'] = DataLoader(
                ImageDataset(dataset.query, transform=transform_test),
                batch_size=test_batch, shuffle=False, num_workers=workers,
                pin_memory=pin_memory, drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                ImageDataset(dataset.gallery, transform=transform_test),
                batch_size=test_batch, shuffle=False, num_workers=workers,
                pin_memory=pin_memory, drop_last=False
            )

        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.train_names))
        print("  # train datasets : {}".format(len(self.train_names)))
        print("  # train ids      : {}".format(self.num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  # train cameras  : {}".format(self.num_train_cams))
        print("  test names       : {}".format(self.test_names))
        print("  *****************************************")
        print("\n")