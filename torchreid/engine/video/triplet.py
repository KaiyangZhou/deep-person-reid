from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine.image import ImageTripletEngine
from torchreid.engine.video import VideoSoftmaxEngine


class VideoTripletEngine(ImageTripletEngine, VideoSoftmaxEngine):

    def __init__(self, dataset, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, scheduler=None, use_cpu=False,
                 label_smooth=True, pooling_method='avg'):
        super(VideoTripletEngine, self).__init__(dataset, model, optimizer, margin=margin,
                                                 weight_t=weight_t, weight_x=weight_x,
                                                 scheduler=scheduler, use_cpu=use_cpu,
                                                 label_smooth=label_smooth)
        self.pooling_method = pooling_method