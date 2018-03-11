from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['DenseNet121']

class DenseNet121(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.classifier.weight.data.uniform_(-1, 1)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if not self.training:
            return x
        x = self.classifier(x)
        return x