from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss, TripletLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageTripletEngine(engine.Engine):
    """Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
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
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
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
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """   

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, scheduler=None, use_cpu=False,
                 label_smooth=True):
        super(ImageTripletEngine, self).__init__(datamanager, model, optimizer, scheduler, use_cpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(self, epoch, trainloader, fixbase=False, open_layers=None, print_freq=10):
        """Trains the model for one epoch on source datasets using hard mining triplet loss.

        Args:
            epoch (int): current epoch.
            trainloader (Dataloader): training dataloader.
            fixbase (bool, optional): whether to fix base layers. Default is False.
            open_layers (str or list, optional): layers open for training.
            print_freq (int, optional): print frequency. Default is 10.
        """
        losses_t = AverageMeter()
        losses_x = AverageMeter()
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
            outputs, features = self.model(imgs)
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)
            loss = self.weight_t * loss_t + self.weight_x * loss_x
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq==0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Triplet {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                      'Softmax {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                      epoch + 1, batch_idx + 1, len(trainloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss_t=losses_t,
                      loss_x=losses_x,
                      acc=accs))
            
            end = time.time()

        if (self.scheduler is not None) and (not fixbase):
            self.scheduler.step()