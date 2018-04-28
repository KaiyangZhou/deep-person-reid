from __future__ import absolute_import

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
    """Fire Module"""
    def __init__(self, in_channels, s1_channels, e1_channels, e3_channels):
        super(FireModule, self).__init__()
        self.squeeze = ConvBlock(in_channels, s1_channels, 1)
        self.expand = ExpandLayer(s1_channels, e1_channels, e3_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.expand(x)
        return x

class SqueezeNet(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SqueezeNet, self).__init__()
        self.loss = loss

        self.conv1 = ConvBlock(3, 96, 7, s=2, p=2)
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)
        self.fire9 = FireModule(512, 64, 256, 256)
        self.conv10 = ConvBlock(512, 1000, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.fire9(x)
        x = self.conv10(x)
        x = F.avg_pool2d(x, x.size()[2:])
        return x

if __name__ == '__main__':
    model = SqueezeNet(10)
    model.eval()
    x = torch.rand(1, 3, 256, 128)
    with torch.no_grad():
        y = model(x)
        print "output size {}".format(y.size())