from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
from torch.autograd import Variable
import math

__all__ = ['MobileNetV2_IFN']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution (bias discarded) + batch normalization + relu6.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
                 to output channels (default: 1).
    """
    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, IN=False):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)#pw
        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)#dw
        #pw-linear 1x1 conv is like linear

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.IN = None

        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)




    def forward(self, x):
        m = self.conv1(x)
        m = self.dwconv2(m)
        m = self.conv3(m)


        if self.use_residual:
            out = x + m
        else:
            out = m

        if self.IN is not None:
            return self.IN(out)
        else:
            return out

class MobileNetV2_IFN(nn.Module):
    """MobileNetV2

    Reference:
    Sandler et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
    """
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(MobileNetV2_IFN, self).__init__()
        self.loss = loss

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
		
        # self.conv1 = ConvBlock(3, 32, 3, s=2, p=1)

        self.block2 = Bottleneck(32, 16, 1, 1,IN=True)
        self.block3 = nn.Sequential(
            Bottleneck(16, 24, 6, 2,IN=True),
            Bottleneck(24, 24, 6, 1,IN=True),
        )
        self.block4 = nn.Sequential(
            Bottleneck(24, 32, 6, 2,IN=True),
            Bottleneck(32, 32, 6, 1,IN=True),
            Bottleneck(32, 32, 6, 1,IN=True),
        )
        self.block5 = nn.Sequential(
            Bottleneck(32, 64, 6, 2,IN=True),
            Bottleneck(64, 64, 6, 1,IN=True),
            Bottleneck(64, 64, 6, 1,IN=True),
            Bottleneck(64, 64, 6, 1,IN=True),
        )
        self.block6 = nn.Sequential(
            Bottleneck(64, 96, 6, 1,IN=True),
            Bottleneck(96, 96, 6, 1,IN=True),
            Bottleneck(96, 96, 6, 1,IN=True),
        )
        self.block7 = nn.Sequential(
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
        )
        self.block8 = Bottleneck(160, 320, 6, 1)
        self.conv9 = ConvBlock(320, 1280, 1)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fn = nn.BatchNorm1d(1280)
        self.fn.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(1280, num_classes, bias=False)

        self.fn.apply(weights_init_kaiming)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # x = self.conv1(x)
        x = F.relu6(self.in1(self.conv1(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.conv9(x)   #n*1280*8*4
		
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fn(x)
        #
        if not self.training:
            return x

        y = self.classifier(x)


        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))