from __future__ import absolute_import
from __future__ import division

import torch


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize, gamma=0.1):
    lr = base_lr * (gamma ** (epoch // stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False