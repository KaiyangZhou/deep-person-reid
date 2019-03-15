from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler, RandomSampler


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains a list of (img_path, pid, camid).
        batch_size (int): number of examples in a batch.
        num_instances (int): number of instances per identity in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(data_source, train_sampler, train_batch_size, num_instances, **kwargs):
    """Builds a training sampler.

    Args:
        data_source (list): contains a list of (img_path, pid, camid).
        train_sampler (str): sampler name (default: RandomSampler).
        train_batch_size (int): training batch size.
        num_instances (int): number of instances per identity in a batch (for RandomIdentitySampler).
    """
    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, train_batch_size, num_instances)
    
    else:
        sampler = RandomSampler(data_source)

    return sampler