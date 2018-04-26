from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
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
        x = F.upsample(x, (x.size(2)*2, x.size(3)*2), mode='bilinear', align_corners=True)
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
    """Hard Attention (Sec. 3.1.II)
    Output: num_regions*2 transformation parameters (i.e. t_x, t_y).
    """
    def __init__(self, in_channels, num_regions):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, num_regions*2)
        self.num_regions = num_regions

    def init_params(self):
        self.fc.weight.data.zero_()
        # TODO
        #self.fc.bias.data.copy_(torch.tensor([BLAH BLAH], dtype=torch.float))

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        theta = F.tanh(self.fc(x))
        theta = theta.view(-1, self.num_regions, 2)
        return theta

class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""
    def __init__(self, in_channels, num_regions):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels, num_regions)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta

class HACNN(nn.Module):
    """
    Harmonious Attention Convolutional Neural Network

    Reference:
    Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.
    """
    def __init__(self, num_classes, loss={'xent'}, num_regions=4, nchannels=[32, 64, 96], feat_dim=512, **kwargs):
        super(HACNN, self).__init__()
        self.loss = loss
        self.num_regions = num_regions
        self.init_scale_factors(num_regions)

        self.conv = ConvBlock(3, 32, 3, s=2, p=1)

        # construct Inception + HarmAttn blocks
        # output channel of InceptionA is out_channels*4
        # output channel of InceptionB is out_channels*2+in_channels
        # ============== Block 1 ==============
        self.inception1 = nn.Sequential(
            InceptionA(32, nchannels[0]),
            InceptionB(nchannels[0]*4, nchannels[0]),
        )
        self.ha1 = HarmAttn(nchannels[0]*6, num_regions)
        self.local_conv1 = InceptionB(32, nchannels[0])

        # ============== Block 2 ==============
        self.inception2 = nn.Sequential(
            InceptionA(nchannels[0]*6, nchannels[1]),
            InceptionB(nchannels[1]*4, nchannels[1]),
        )
        self.ha2 = HarmAttn(nchannels[1]*6, num_regions)
        self.local_conv2 = InceptionB(nchannels[0]*2+32, nchannels[1])

        # ============== Block 3 ==============
        self.inception3 = nn.Sequential(
            InceptionA(nchannels[1]*6, nchannels[2]),
            InceptionB(nchannels[2]*4, nchannels[2]),
        )
        self.ha3 = HarmAttn(nchannels[2]*6, num_regions)
        self.local_conv3 = InceptionB(nchannels[1]*2+nchannels[0]*2+32, nchannels[2])

        # feature embedding layers
        self.fc_global = nn.Sequential(
            nn.Linear(nchannels[2]*6, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        self.fc_local = nn.Sequential(
            nn.Linear((nchannels[2]*2+nchannels[1]*2+nchannels[0]*2+32)*num_regions, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        self.classifier_global = nn.Linear(feat_dim, num_classes)
        self.classifier_local = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim

    def init_scale_factors(self, num_regions):
        self.scale_factors = []
        for region_idx in range(num_regions):
            # TODO: initialize scale factors
            scale_factors = torch.tensor([[1, 0], [0, 1]]).float()
            self.scale_factors.append(scale_factors)

    def stn(self, x, theta):
        """Perform spatial transform
        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transform theta from (batch, 2) to (batch, 2, 3),
        which includes (s_w, s_h)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,:,:2] = scale_factors
        theta[:,:,-1] = theta_i
        return theta

    def forward(self, x):
        # input size (3, 160, 64)
        x = self.conv(x)

        # ============== Block 1 ==============
        # global branch
        x1 = self.inception1(x)
        x1_attn, x1_theta = self.ha1(x1)
        x1_out = x1 * x1_attn
        # local branch
        x1_local = []
        for region_idx in range(self.num_regions):
            x1_theta_i = x1_theta[:,region_idx,:]
            x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
            x1_trans_i = self.stn(x, x1_theta_i)
            # TODO: reduce size of x1_trans_i to (24, 28)
            sys.exit()
            x1_local_i = self.local_conv1(x1_trans_i)
            x1_local.append(x1_local_i)

        # ============== Block 2 ==============
        # Block 2
        # global branch
        x2 = self.inception2(x1_out)
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2 * x2_attn
        # local branch

        # ============== Block 3 ==============
        # Block 3
        # global branch
        x3 = self.inception3(x2_out)
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn
        # local branch

        x_global = F.avg_pool2d(x3_out, x3_out.size()[2:]).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)

        if not self.training:
            return x_global

        prelogits = self.classifier_global(x_global)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, x_global
        elif self.loss == {'cent'}:
            return prelogits, x_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

if __name__ == '__main__':
    pass