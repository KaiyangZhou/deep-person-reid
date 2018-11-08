from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize=20, gamma=0.1,
                         linear_decay=False, final_lr=0, max_epoch=100):
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done * final_lr + (1. - frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma ** (epoch // stepsize))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def open_all_layers(model):
    """
    Open all layers in model for training.

    Args:
    - model (nn.Module): neural net model.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping
    other layers frozen.

    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layer names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    for layer in open_layers:
        assert hasattr(model, layer), "'{}' is not an attribute of the model, please provide the correct name".format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param