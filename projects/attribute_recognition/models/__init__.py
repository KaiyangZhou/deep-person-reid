from __future__ import absolute_import

from .osnet import *

__model_factory = {
    'osnet_avgpool': osnet_avgpool,
    'osnet_maxpool': osnet_maxpool
}


def build_model(name, num_classes, pretrained=True, device='cuda'):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError
    return __model_factory[name](
        num_classes=num_classes, pretrained=pretrained, use_gpu=device.startswith('cuda')
    )
