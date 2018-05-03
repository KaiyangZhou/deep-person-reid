from __future__ import absolute_import

from .ResNet import *
from .DenseNet import *
from .MuDeep import *
from .HACNN import *
from .SqueezeNet import *
from .MobileNet import *
from .ShuffleNet import *
from .Xception import *
from .Inception import *
from .SEResNet import *

__factory = {
    'resnet50': ResNet50,
    'densenet121': DenseNet121,
    'resnet50m': ResNet50M,
    'mudeep': MuDeep,
    'hacnn': HACNN,
    'squeezenet': SqueezeNet,
    'mobilenet': MobileNetV2,
    'shufflenet': ShuffleNet,
    'xception': Xception,
    'inceptionv4': InceptionV4ReID,
    'seresnet50': SEResNet50,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)