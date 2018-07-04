from __future__ import absolute_import
from __future__ import division

import torch


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize, gamma=0.1):
    lr = base_lr * (gamma ** (epoch // stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr