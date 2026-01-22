from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss


class ImageSoftmaxNASEngine(Engine):

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        device='cuda',
        label_smooth=True,
        mc_iter=1,
        init_lmda=1.,
        min_lmda=1.,
        lmda_decay_step=20,
        lmda_decay_rate=0.5,
        fixed_lmda=False
    ):
        super(ImageSoftmaxNASEngine, self).__init__(datamanager, device)
        self.mc_iter = mc_iter
        self.init_lmda = init_lmda
        self.min_lmda = min_lmda
        self.lmda_decay_step = lmda_decay_step
        self.lmda_decay_rate = lmda_decay_rate
        self.fixed_lmda = fixed_lmda

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            device=self.device,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.to(self.device)
        pids = pids.to(self.device)

        # softmax temporature
        if self.fixed_lmda or self.lmda_decay_step == -1:
            lmda = self.init_lmda
        else:
            lmda = self.init_lmda * self.lmda_decay_rate**(
                self.epoch // self.lmda_decay_step
            )
            if lmda < self.min_lmda:
                lmda = self.min_lmda

        for k in range(self.mc_iter):
            outputs = self.model(imgs, lmda=lmda)
            loss = self.compute_loss(self.criterion, outputs, pids)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_dict
