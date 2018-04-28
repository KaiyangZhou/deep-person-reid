from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['DaRe']

class DaRe(nn.Module):
    def __init__(self, num_classes=0, loss={'xent'}, w_init=0.1, **kwargs):
        super(DaRe, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])

        # construct four convolutional blocks
        self.conv_block1 = nn.Sequential(
            base[0], base[1], base[2], base[3],
            base[4][0], base[4][1],
            base[4][2].conv1, base[4][2].bn1, base[4][2].conv2, base[4][2].bn2,
        )
        self.linears1 = nn.Sequential(nn.Linear(64, 1204), nn.BatchNorm1d(1204), nn.ReLU(), nn.Linear(1204, 128))

        self.conv_block2 = nn.Sequential(
            base[4][2].conv3, base[4][2].bn3, base[4][2].relu,
            base[5][0], base[5][1], base[5][2],
            base[5][3].conv1, base[5][3].bn1, base[5][3].conv2, base[5][3].bn2,
        )
        self.linears2 = nn.Sequential(nn.Linear(128, 1204), nn.BatchNorm1d(1204), nn.ReLU(), nn.Linear(1204, 128))

        self.conv_block3 = nn.Sequential(
            base[5][3].conv3, base[5][3].bn3, base[5][3].relu,
            base[6][0], base[6][1], base[6][2], base[6][3], base[6][4],
            base[6][5].conv1, base[6][5].bn1, base[6][5].conv2, base[6][5].bn2,
        )
        self.linears3 = nn.Sequential(nn.Linear(256, 1204), nn.BatchNorm1d(1204), nn.ReLU(), nn.Linear(1204, 128))

        self.conv_block4 = nn.Sequential(
            base[6][5].conv3, base[6][5].bn3, base[6][5].relu,
            base[7][0], base[7][1],
            base[7][2].conv1, base[7][2].bn1, base[7][2].conv2, base[7][2].bn2,
        )
        self.linears4 = nn.Sequential(nn.Linear(512, 1204), nn.BatchNorm1d(1204), nn.ReLU(), nn.Linear(1204, 128))

        # fusion weights for four stages
        self.w1 = nn.Parameter(torch.ones(1) * w_init)
        self.w2 = nn.Parameter(torch.ones(1) * w_init)
        self.w3 = nn.Parameter(torch.ones(1) * w_init)
        self.w4 = nn.Parameter(torch.ones(1) * w_init)

        self.classifier = nn.Linear(128, num_classes)
        self.feat_dim = 128 # feature dimension

    def forward(self, x):
        x1 = self.conv_block1(x)
        x1_feat = F.avg_pool2d(x1, x1.size()[2:]).view(x1.size(0), x1.size(1))
        x1_feat = self.linears1(x1_feat)

        x2 = self.conv_block2(x1)
        x2_feat = F.avg_pool2d(x2, x2.size()[2:]).view(x2.size(0), x2.size(1))
        x2_feat = self.linears2(x2_feat)

        x3 = self.conv_block3(x2)
        x3_feat = F.avg_pool2d(x3, x3.size()[2:]).view(x3.size(0), x3.size(1))
        x3_feat = self.linears3(x3_feat)

        x4 = self.conv_block4(x3)
        x4_feat = F.avg_pool2d(x4, x4.size()[2:]).view(x4.size(0), x4.size(1))
        x4_feat = self.linears4(x4_feat)

        fusion_feat = x1_feat * self.w1 + x2_feat * self.w2 + x3_feat * self.w3 + x4_feat * self.w4

        if not self.training:
            return fusion_feat

        prelogits = self.classifier(fusion_feat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, (fusion_feat, x1_feat, x2_feat, x3_feat, x4_feat)
        elif self.loss == {'cent'}:
            return prelogits, fusion_feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
