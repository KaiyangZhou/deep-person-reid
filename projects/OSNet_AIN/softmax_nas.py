from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid import metrics
from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss


class ImageSoftmaxNASEngine(Engine):

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=False,
        label_smooth=True,
        mc_iter=1,
        init_lmda=1.,
        min_lmda=1.,
        lmda_decay_step=20,
        lmda_decay_rate=0.5,
        fixed_lmda=False
    ):
        super(ImageSoftmaxNASEngine, self
              ).__init__(datamanager, model, optimizer, scheduler, use_gpu)
        self.mc_iter = mc_iter
        self.init_lmda = init_lmda
        self.min_lmda = min_lmda
        self.lmda_decay_step = lmda_decay_step
        self.lmda_decay_rate = lmda_decay_rate
        self.fixed_lmda = fixed_lmda

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(
        self,
        epoch,
        max_epoch,
        writer,
        fixbase_epoch=0,
        open_layers=None,
        print_freq=10
    ):
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            # softmax temporature
            if self.fixed_lmda or self.lmda_decay_step == -1:
                lmda = self.init_lmda
            else:
                lmda = self.init_lmda * self.lmda_decay_rate**(
                    epoch // self.lmda_decay_step
                )
                if lmda < self.min_lmda:
                    lmda = self.min_lmda

            for k in range(self.mc_iter):
                outputs = self.model(imgs, lmda=lmda)
                loss = self._compute_loss(self.criterion, outputs, pids)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (
                    num_batches - (batch_idx+1) + (max_epoch -
                                                   (epoch+1)) * num_batches
                )
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'Lr {lr:.6f}\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accs,
                        lr=self.optimizer.param_groups[0]['lr'],
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch*num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Data', data_time.avg, n_iter)
                writer.add_scalar('Train/Loss', losses.avg, n_iter)
                writer.add_scalar('Train/Acc', accs.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
