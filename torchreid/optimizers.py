from __future__ import absolute_import

import torch


def init_optimizer(params,
                   optim='adam',
                   lr=0.003,
                   weight_decay=5e-4,
                   momentum=0.9, # momentum factor for sgd and rmsprop
                   sgd_dampening=0, # sgd's dampening for momentum
                   sgd_nesterov=False, # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99, # rmsprop's smoothing constant
                   adam_beta1=0.9, # exponential decay rate for adam's first moment
                   adam_beta2=0.999 # # exponential decay rate for adam's second moment
                   ):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))
    
    elif optim == 'amsgrad':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)
    
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)
    
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)
    
    else:
        raise ValueError("Unsupported optimizer: {}".format(optim))