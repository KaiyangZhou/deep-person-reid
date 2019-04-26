from __future__ import absolute_import
from __future__ import print_function

import torch


AVAI_SCH = ['single_step', 'multi_step']


def build_lr_scheduler(optimizer, lr_scheduler, stepsize, gamma=0.1):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str): learning rate scheduler method. Currently supports
            "single_step" and "multi_step".
        stepsize (int or list): step size to decay learning rate. When ``lr_scheduler`` is
            "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list.
        gamma (float, optional): decay rate. Default is 0.1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))
    
    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]
        
        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    return scheduler