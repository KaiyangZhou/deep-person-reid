from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn


def init_optimizer(
    model,
    optim='adam',  # optimizer choices
    lr=0.003,  # learning rate
    weight_decay=5e-4,  # weight decay
    momentum=0.9,  # momentum factor for sgd and rmsprop
    sgd_dampening=0,  # sgd's dampening for momentum
    sgd_nesterov=False,  # whether to enable sgd's Nesterov momentum
    rmsprop_alpha=0.99,  # rmsprop's smoothing constant
    adam_beta1=0.9,  # exponential decay rate for adam's first moment
    adam_beta2=0.999,  # # exponential decay rate for adam's second moment
    staged_lr=False,  # different lr for different layers
    new_layers=None,  # new layers use the default lr, while other layers's lr is scaled by base_lr_mult
    base_lr_mult=0.1,  # learning rate multiplier for base layers
    ):

    if staged_lr:
        assert new_layers is not None
        base_params = []
        base_layers = []
        new_params = []
        if isinstance(model, nn.DataParallel):
            model = model.module
        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
        param_groups = [
            {'params': base_params, 'lr': lr * base_lr_mult},
            {'params': new_params},
        ]
        print('Use staged learning rate')
        print(
            '* Base layers (initial lr = {}): {}'.format(lr * base_lr_mult, base_layers)
        )
        print('* New layers (initial lr = {}): {}'.format(lr, new_layers))

    else:
        param_groups = model.parameters()

    print('Initializing optimizer: {}'.format(optim))

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))

    return optimizer
