from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['MobileNetV2']


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
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwconv2(m)
        m = self.conv3(m)
        if self.use_residual:
            return x + m
        else:
            return m



class MobileNetV2(nn.Module):
    """
    MobileNetV2

    Reference:
    Sandler et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
    """
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(MobileNetV2, self).__init__()
        self.loss = loss

        self.conv1 = ConvBlock(3, 32, 3, s=2, p=1)
        self.block2 = Bottleneck(32, 16, 1, 1)
        self.block3 = nn.Sequential(
            Bottleneck(16, 24, 6, 2),
            Bottleneck(24, 24, 6, 1),
        )
        self.block4 = nn.Sequential(
            Bottleneck(24, 32, 6, 2),
            Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 32, 6, 1),
        )
        self.block5 = nn.Sequential(
            Bottleneck(32, 64, 6, 2),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 64, 6, 1),
        )
        self.block6 = nn.Sequential(
            Bottleneck(64, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
            Bottleneck(96, 96, 6, 1),
        )
        self.block7 = nn.Sequential(
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
        )
        self.block8 = Bottleneck(160, 320, 6, 1)
        self.conv9 = ConvBlock(320, 1280, 1)
        self.classifier = nn.Linear(1280, num_classes)
        self.feat_dim = 1280

    def forward(self, x):
        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.conv9(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = F.dropout(x, training=self.training)

        if not self.training:
            return x

        y = self.classifier(x)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))