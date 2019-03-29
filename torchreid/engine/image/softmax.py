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
    """Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_cpu (bool, optional): use cpu. Default is False.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torch
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)
        
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(self, epoch, trainloader, fixbase=False, open_layers=None, print_freq=10):
        """Trains the model for one epoch on source datasets using softmax loss.

        Args:
            epoch (int): current epoch.
            trainloader (Dataloader): training dataloader.
            fixbase (bool, optional): whether to fix base layers. Default is False.
            open_layers (str or list, optional): layers open for training.
            print_freq (int, optional): print frequency. Default is 10.
        """
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

        if (self.scheduler is not None) and (not fixbase):
            self.scheduler.step()