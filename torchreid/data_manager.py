from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, VideoDataset
from .datasets import init_imgreid_dataset, init_vidreid_dataset
from .transforms import build_transforms
from .samplers import build_train_sampler


class BaseDataManager(object):

    def __init__(
        self,
        use_gpu,
        source_names,
        target_names,
        root='data',
        split_id=0,
        height=256,
        width=128,
        combineall=False, # combine all data in a dataset for training
        train_batch_size=32,
        test_batch_size=100,
        workers=4,
        train_sampler='',
        random_erase=False, # use random erasing for data augmentation
        color_jitter=False, # randomly change the brightness, contrast and saturation
        color_aug=False, # randomly alter the intensities of RGB channels
        num_instances=4, # number of instances per identity (for RandomIdentitySampler)
        **kwargs
        ):
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.combineall = combineall
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug
        self.num_instances = num_instances

        transform_train, transform_test = build_transforms(
            self.height, self.width,
            random_erase=self.random_erase,
            color_jitter=self.color_jitter,
            color_aug=self.color_aug
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """Return trainloader and testloader dictionary"""
        return self.trainloader, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """Return query and gallery, each containing a list of (img_path, pid, camid)"""
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):

    def __init__(
        self,
        use_gpu,
        source_names,
        target_names,
        cuhk03_labeled=False, # use cuhk03's labeled or detected images
        cuhk03_classic_split=False, # use cuhk03's classic split or 767/700 split
        market1501_500k=False, # add 500k distractors to the gallery set for market1501
        **kwargs
        ):
        super(ImageDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)
        self.cuhk03_labeled = cuhk03_labeled
        self.cuhk03_classic_split = cuhk03_classic_split
        self.market1501_500k = market1501_500k

        print('=> Initializing train (source) datasets')
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_imgreid_dataset(
                root=self.root,
                name=name,
                split_id=self.split_id,
                combineall=self.combineall,
                cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split,
                market1501_500k=self.market1501_500k
            )

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        self.train_sampler = build_train_sampler(
            train,
            self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )

        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train),
            sampler=self.train_sampler,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Initializing test (target) datasets')
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        
        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root,
                name=name,
                split_id=self.split_id,
                combineall=self.combineall,
                cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split,
                market1501_500k=self.market1501_500k
            )

            self.testloader_dict[name]['query'] = DataLoader(
                ImageDataset(dataset.query, transform=self.transform_test),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                ImageDataset(dataset.gallery, transform=self.transform_test),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train names      : {}'.format(self.source_names))
        print('  # train datasets : {}'.format(len(self.source_names)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(train)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test names       : {}'.format(self.target_names))
        print('  *****************************************')
        print('\n')


class VideoDataManager(BaseDataManager):

    def __init__(
        self,
        use_gpu,
        source_names,
        target_names,
        seq_len=15,
        sample_method='evenly',
        image_training=True, # train the video-reid model with images rather than tracklets
        **kwargs
        ):
        super(VideoDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.image_training = image_training

        print('=> Initializing train (source) datasets')
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id, combineall=self.combineall)

            for img_paths, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                if image_training:
                    # decompose tracklets into images
                    for img_path in img_paths:
                        train.append((img_path, pid, camid))
                else:
                    train.append((img_paths, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        self.train_sampler = build_train_sampler(
            train, self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )

        if image_training:
            # each batch has image data of shape (batch, channel, height, width)
            self.trainloader = DataLoader(
                ImageDataset(train, transform=self.transform_train),
                sampler=self.train_sampler,
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=True
            )
        
        else:
            # each batch has image data of shape (batch, seq_len, channel, height, width)
            self.trainloader = DataLoader(
                VideoDataset(
                    train,
                    seq_len=self.seq_len,
                    sample_method=self.sample_method,
                    transform=self.transform_train
                ),
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=True
            )
            raise NotImplementedError('This requires a new trainer')

        print('=> Initializing test (target) datasets')
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in target_names}

        for name in self.target_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id, combineall=self.combineall,)

            self.testloader_dict[name]['query'] = DataLoader(
                VideoDataset(
                    dataset.query,
                    seq_len=self.seq_len,
                    sample_method=self.sample_method,
                    transform=self.transform_test
                ),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                VideoDataset(
                    dataset.gallery,
                    seq_len=self.seq_len,
                    sample_method=self.sample_method,
                    transform=self.transform_test
                ),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train names       : {}'.format(self.source_names))
        print('  # train datasets  : {}'.format(len(self.source_names)))
        print('  # train ids       : {}'.format(self.num_train_pids))
        if self.image_training:
            print('  # train images   : {}'.format(len(train)))
        else:
            print('  # train tracklets: {}'.format(len(train)))
        print('  # train cameras   : {}'.format(self.num_train_cams))
        print('  test names        : {}'.format(self.target_names))
        print('  *****************************************')
        print('\n')