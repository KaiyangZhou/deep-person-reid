from __future__ import absolute_import
from __future__ import print_function

import torch


AVAI_SCH = ['single_step', 'multi_step']


def build_lr_scheduler(optimizer, lr_scheduler, stepsize, gamma=0.1):
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))

    print('Initializing lr_scheduler: {}'.format(lr_scheduler))
    
    if lr_scheduler == 'single_step':
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