from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['SqueezeNet']


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ExpandLayer(nn.Module):
    def __init__(self, in_channels, e1_channels, e3_channels):
        super(ExpandLayer, self).__init__()
        self.conv11 = ConvBlock(in_channels, e1_channels, 1)
        self.conv33 = ConvBlock(in_channels, e3_channels, 3, p=1)

    def forward(self, x):
        x11 = self.conv11(x)
        x33 = self.conv33(x)
        x = torch.cat([x11, x33], 1)
        return x


class FireModule(nn.Module):
    """
    Args:
        in_channels (int): number of input channels.
        s1_channels (int): number of 1-by-1 filters for squeeze layer.
        e1_channels (int): number of 1-by-1 filters for expand layer.
        e3_channels (int): number of 3-by-3 filters for expand layer.

    Number of output channels from FireModule is e1_channels + e3_channels.
    """
    def __init__(self, in_channels, s1_channels, e1_channels, e3_channels):
        super(FireModule, self).__init__()
        self.squeeze = ConvBlock(in_channels, s1_channels, 1)
        self.expand = ExpandLayer(s1_channels, e1_channels, e3_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.expand(x)
        return x


class SqueezeNet(nn.Module):
    """SqueezeNet

    Reference:
    Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and< 0.5 MB model size. arXiv:1602.07360.
    """
    def __init__(self, num_classes, loss={'xent'}, bypass=True, **kwargs):
        super(SqueezeNet, self).__init__()
        self.loss = loss
        self.bypass = bypass

        self.conv1 = ConvBlock(3, 96, 7, s=2, p=2)
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)
        self.fire9 = FireModule(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(512, num_classes, 1)
        
        self.feat_dim = num_classes

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.max_pool2d(x1, 3, stride=2)
        x2 = self.fire2(x1)
        x3 = self.fire3(x2)
        if self.bypass:
            x3 = x3 + x2
        x4 = self.fire4(x3)
        x4 = F.max_pool2d(x4, 3, stride=2)
        x5 = self.fire5(x4)
        if self.bypass:
            x5 = x5 + x4
        x6 = self.fire6(x5)
        x7 = self.fire7(x6)
        if self.bypass:
            x7 = x7 + x6
        x8 = self.fire8(x7)
        x8 = F.max_pool2d(x8, 3, stride=2)
        x9 = self.fire9(x8)
        if self.bypass:
            x9 = x9 + x8
        x9 = F.dropout(x9, training=self.training)
        x10 = F.relu(self.conv10(x9))
        f = F.avg_pool2d(x10, x10.size()[2:]).view(x10.size(0), -1)

        if not self.training:
            return f

        if self.loss == {'xent'}:
            return f
        elif self.loss == {'xent', 'htri'}:
            return f, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))