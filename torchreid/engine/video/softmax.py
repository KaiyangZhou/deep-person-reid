from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine.image import ImageSoftmaxEngine


class VideoSoftmaxEngine(ImageSoftmaxEngine):

    def __init__(self, datamanager, model, optimizer, scheduler=None,
                 use_cpu=False, label_smooth=True, pooling_method='avg'):
        super(VideoSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler=scheduler,
                                                 use_cpu=use_cpu, label_smooth=label_smooth)
        self.pooling_method = pooling_method

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        if imgs.dim() == 5:
            # b: batch size
            # s: sqeuence length
            # c: channel depth
            # h: height
            # w: width
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b*s, c, h, w)
            pids = pids.view(b, 1).expand(b, s)
            pids = pids.contiguous().view(b*s)
        return imgs, pids

    def _extract_features(self, input):
        self.model.eval()
        # b: batch size
        # s: sqeuence length
        # c: channel depth
        # h: height
        # w: width
        b, s, c, h, w = input.size()
        input = input.view(b*s, c, h, w)
        features = self.model(input)
        features = features.view(b, s, -1)
        if self.pooling_method == 'avg':
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
        return features