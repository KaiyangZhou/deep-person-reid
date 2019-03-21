from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageSoftmaxEngine(engine.Engine):

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)
        
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(self, epoch, trainloader, fixbase=False, open_layers=None, print_freq=10):
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()

        if fixbase and (open_layers is not None):
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self._compute_loss(self.criterion, outputs, pids)
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq==0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                      epoch + 1, batch_idx + 1, len(trainloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accs))
            
            end = time.time()