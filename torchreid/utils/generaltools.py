from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import random
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)