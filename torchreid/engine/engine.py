from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import time
import datetime
import numpy as np

import torch
import torch.nn as nn

import torchreid
from torchreid.utils import AverageMeter, visualize_ranked_results, save_checkpoint
from torchreid.losses import DeepSupervision
from torchreid import metrics


class Engine(object):

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_cpu=False):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and not use_cpu)

        # check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError('optimizer must be an instance of torch.optim.Optimizer')

    def run(self, max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, save_dir='log', test_only=False, print_freq=10,
            dist_metric='euclidean', visrank=False, visrank_topk=20,
            use_metric_cuhk03=False, ranks=[1, 5, 10, 20]):
        trainloader, testloader = self.datamanager.return_dataloaders()

        if test_only:
            self.test(testloader, dist_metric, visrank, visrank_topk,
                      save_dir, use_metric_cuhk03, ranks)
            return

        time_start = time.time()
        print('=> Start training')

        if fixbase_epoch>0 and (open_layers is not None):
            print('Pretrain open layers ({}) for {} epochs'.format(open_layers, fixbase_epoch))
            for epoch in range(fixbase_epoch):
                self.train(epoch, trainloader, fixbase=True, open_layers=open_layers,
                           print_freq=print_freq)
            print('Done. From now on all layers are open to train for {} epochs'.format(max_epoch))

        for epoch in range(start_epoch, max_epoch):
            self.train(epoch, trainloader, print_freq=print_freq)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if (epoch+1)>start_eval and eval_freq>0 and (epoch+1)%eval_freq==0 and (epoch+1)!=max_epoch:
                rank1 = self.test(testloader, dist_metric, visrank, visrank_topk,
                                  save_dir, use_metric_cuhk03, ranks)
                self._save_checkpoint(epoch, rank1, save_dir)

        print('=> Final test')
        rank1 = self.test(
            testloader, dist_metric, visrank, visrank_topk,
            save_dir, use_metric_cuhk03, ranks
        )
        self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))

    def train(self):
        raise NotImplementedError

    def test(self, testloader, dist_metric='euclidean', visrank=False, visrank_topk=20,
             save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20]):
        target_names = list(testloader.keys())
        
        for name in target_names:
            print('Evaluate {}'.format(name))
            queryloader = testloader[name]['query']
            galleryloader = testloader[name]['gallery']
            rank1 = self._evaluate(
                name, queryloader, galleryloader, dist_metric, visrank,
                visrank_topk, save_dir, use_metric_cuhk03, ranks
            )
        
        return rank1

    @torch.no_grad()
    def _evaluate(self, dataset_name, queryloader, galleryloader, dist_metric,
                  visrank, visrank_topk, save_dir, use_metric_cuhk03, ranks):
        batch_time = AverageMeter()

        self.model.eval()

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = [], [], []
        for batch_idx, data in enumerate(queryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, data in enumerate(galleryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                save_dir=osp.join(save_dir, 'ranked_results', dataset_name),
                topk=visrank_topk
            )

        return cmc[0]

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)