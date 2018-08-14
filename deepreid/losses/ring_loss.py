from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        loss = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return loss