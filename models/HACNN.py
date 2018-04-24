from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['HACNN']

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

class InceptionA(nn.Module):
    """InceptionA (https://github.com/Cysu/dgd_person_reid)"""
    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        self.stream1 = ConvBlock(in_channels, out_channels, 1)
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels, 3, p=1),
        )
        self.stream3 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels, 3, p=1),
            ConvBlock(out_channels, out_channels, 3, p=1),
        )
        self.stream4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, out_channels, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y

class InceptionB(nn.Module):
    """InceptionB (https://github.com/Cysu/dgd_person_reid)"""
    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels, 3, s=2, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels, 3, p=1),
            ConvBlock(out_channels, out_channels, 3, s=2, p=1),
        )
        self.stream3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(x, (x.size(2)*2, x.size(3)*2), mode='bilinear')
        # scaling conv
        x = self.conv2(x)
        return x

class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels/reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels/reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv2(self.conv1(x))
        return x

class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    Aim: Spatial Attention + Channel Attention
    Output: attention maps with shape identical to input.
    """
    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = F.sigmoid(self.conv(y))
        return y

class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""
    def __init__(self):
        super(HardAttn, self).__init__()

    def forward(self, x):
        raise NotImplementedError

class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""
    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        return y_soft_attn

class HACNN(nn.Module):
    """
    Harmonious Attention Convolutional Neural Network

    Reference:
    Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.
    """
    def __init__(self, num_classes, loss={'xent'}, widths=[32, 64, 96], embed_dim=512, **kwargs):
        super(HACNN, self).__init__()
        self.loss = loss
        self.conv = ConvBlock(3, 32, 3, s=2, p=1)
        
        # construct Inception + HarmAttn blocks
        # output channel of InceptionA is out_channels*4
        # output channel of InceptionB is out_channels*2+in_channels
        self.inception1 = nn.Sequential(
            InceptionA(32, widths[0]),
            InceptionB(widths[0]*4, widths[0]),
        )
        self.ha1 = HarmAttn(widths[0]*6)

        self.inception2 = nn.Sequential(
            InceptionA(widths[0]*6, widths[1]),
            InceptionB(widths[1]*4, widths[1]),
        )
        self.ha2 = HarmAttn(widths[1]*6)

        self.inception3 = nn.Sequential(
            InceptionA(widths[1]*6, widths[2]),
            InceptionB(widths[2]*4, widths[2]),
        )
        self.ha3 = HarmAttn(widths[2]*6)

        self.fc_global = nn.Sequential(nn.Linear(widths[2]*6, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU())

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.feat_dim = embed_dim

    def forward(self, x):
        # input size (3, 160, 64)
        x = self.conv(x)

        # block 1
        x1 = self.inception1(x)
        x1_attn = self.ha1(x1)
        x1_out = x1 * x1_attn

        # block 2
        x2 = self.inception2(x1_out)
        x2_attn = self.ha2(x2)
        x2_out = x2 * x2_attn

        # block 3
        x3 = self.inception3(x2_out)
        x3_attn = self.ha3(x3)
        x3_out = x3 * x3_attn

        x_global = F.avg_pool2d(x3_out, x3_out.size()[2:]).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)

        if not self.training:
            return x_global

        prelogits = self.classifier(x_global)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, x_global
        elif self.loss == {'cent'}:
            return prelogits, x_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

if __name__ == '__main__':
    import sys
    model = HACNN(10)
    model.eval()
    x = Variable(torch.rand(5, 3, 160, 64))
    print "input size {}".format(x.size())
    y = model(x)
    print "output size {}".format(y.size())