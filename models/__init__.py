from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *
from .MuDeep import *
from .HACNN import *

__factory = {
    'resnet50': ResNet50,
    'densenet121': DenseNet121,
    'resnet50m': ResNet50M,
    'mudeep': MuDeep,
    'hacnn': HACNN,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)