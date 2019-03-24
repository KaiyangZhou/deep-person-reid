from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from torchreid.data.sampler import build_train_sampler
from torchreid.data.transforms import build_transforms
from torchreid.data.datasets import init_image_dataset, init_video_dataset


class DataManager(object):
    """Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        random_erase (bool, optional): use random erasing. Default is False.
        color_jitter (bool, optional): use color jittering. Default is False.
        color_aug (bool, optional): use color augmentation. Default is False.
        use_cpu (bool, optional): use cpu. Default is False.
    """

    def __init__(self, sources=None, targets=None, height=256, width=128, random_erase=False,
                 color_jitter=False, color_aug=False, use_cpu=False):
        self.sources = sources
        self.targets = targets

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            height, width,
            random_erase=random_erase,
            color_jitter=color_jitter,
            color_aug=color_aug
        )

        self.use_gpu = (torch.cuda.is_available() and not use_cpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def return_dataloaders(self):
        """Returns trainloader and testloader."""
        return self.trainloader, self.testloader

    def return_testdataset_by_name(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        return self.testdataset[name]['query'], self.testdataset[name]['gallery']


class ImageDataManager(DataManager):
    """Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        random_erase (bool, optional): use random erasing. Default is False.
        color_jitter (bool, optional): use color jittering. Default is False.
        color_aug (bool, optional): use color augmentation. Default is False.
        use_cpu (bool, optional): use cpu. Default is False.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size (int, optional): number of images in a batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is empty (``RandomSampler``).
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size=32
        )
    """

    def __init__(self, root='', sources=None, targets=None, height=256, width=128, random_erase=False,
                 color_jitter=False, color_aug=False, use_cpu=False, split_id=0, combineall=False,
                 batch_size=32, workers=4, num_instances=4, train_sampler='',
                 cuhk03_labeled=False, cuhk03_classic_split=False, market1501_500k=False):
        
        super(ImageDataManager, self).__init__(sources=sources, targets=targets, height=height, width=width,
                                               random_erase=random_erase, color_jitter=color_jitter,
                                               color_aug=color_aug, use_cpu=use_cpu)
        
        print('=> Loading train (source) dataset')
        trainset = []  
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train            : {}'.format(self.sources))
        print('  # train datasets : {}'.format(len(self.sources)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(trainset)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')


class VideoDataManager(DataManager):
    """Video data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        random_erase (bool, optional): use random erasing. Default is False.
        color_jitter (bool, optional): use color jittering. Default is False.
        color_aug (bool, optional): use color augmentation. Default is False.
        use_cpu (bool, optional): use cpu. Default is False.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size (int, optional): number of *tracklets* in a batch. Default is 3.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is empty (``RandomSampler``).
        seq_len (int, optional): how many images to sample in a tracklet. Default is 15.
        sample_method (str, optional): how to sample images in a tracklet. Default is "evenly".
            Choices are ["evenly", "random", "all"]. "evenly" and "random" sample ``seq_len``
            images in a tracklet while "all" samples all images in a tracklet, thus ``batch_size``
            needs to be set to 1.

    Examples::

        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            batch_size=3,
            seq_len=15,
            sample_method='evenly'
        )
    """

    def __init__(self, root='', sources=None, targets=None, height=256, width=128, random_erase=False,
                 color_jitter=False, color_aug=False, use_cpu=False, split_id=0, combineall=False,
                 batch_size=3, workers=4, num_instances=4, train_sampler=None,
                 seq_len=15, sample_method='evenly'):
        
        super(VideoDataManager, self).__init__(sources=sources, targets=targets, height=height, width=width,
                                               random_erase=random_erase, color_jitter=color_jitter,
                                               color_aug=color_aug, use_cpu=use_cpu)

        print('=> Loading train (source) dataset')
        trainset = []  
        for name in self.sources:
            trainset_ = init_video_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train             : {}'.format(self.sources))
        print('  # train datasets  : {}'.format(len(self.sources)))
        print('  # train ids       : {}'.format(self.num_train_pids))
        print('  # train tracklets : {}'.format(len(trainset)))
        print('  # train cameras   : {}'.format(self.num_train_cams))
        print('  test              : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')