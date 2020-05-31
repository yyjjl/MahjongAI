# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from torch.optim.lr_scheduler import _LRScheduler

import framework.session_manager as manager
from framework.logging import log_info, log_warning
from framework.utils import AverageMeter, Options

from ..utils import add_stats, to_device
from .dataloader import DataSetOptions
from .model import CNNModel, ModelOptions


@dataclass
class MainOptions(manager.ManagerOptions, ModelOptions, DataSetOptions, Options):
    learning_rate: float
    weight_decay: float
    num_epoch_steps: int
    num_epochs: int


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, min_lr, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(self.min_lr, base_lr * self.gamma ** self.last_epoch)
                for base_lr in self.base_lrs]


class SessionManager(manager.SessionManager):
    def __init__(self, options: MainOptions, training=True):
        super().__init__(options, training)

        self.metrics_name = 'accuracy'
        self.num_epoch_steps = options.num_epoch_steps
        self.num_epochs = options.num_epochs

    def build_model(self):
        return CNNModel(self.options)

    def build_optimizer(self):
        options = self.options
        return optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),
                          lr=options.learning_rate, weight_decay=options.weight_decay)

    def on_update(self, stats, prefix='main'):
        super().on_update(stats, prefix)

        if self.training and self.global_step % self.num_epoch_steps == 0:
            self.scheduler.step()
            log_info('Learning Rate: %s', self.scheduler.get_lr())

    def state_dict(self, saved_state):
        saved_state['optimizer'] = self.optimizer.state_dict()
        saved_state['model'] = self.model.state_dict()
        return saved_state

    def load_state_dict(self, saved_state):
        self.optimizer.load_state_dict(saved_state['optimizer'])
        self.model.load_state_dict(saved_state['model'])

    def _setup_scheduler(self):
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.5, min_lr=1e-5,
                                       last_epoch=self.global_step // self.num_epoch_steps - 1)

    def _restore(self):
        super()._restore()
        self._setup_scheduler()

    def _init(self):
        super()._init()
        self._setup_scheduler()

    def train(self, dataset):
        model = self.model
        optimizer = self.optimizer
        device = self.device

        for i in range(self.num_epochs):
            meter = AverageMeter()
            for batch in dataset.iter_train_dataset():
                self.run_batch(model, optimizer, batch, meter, True, device)
                self.on_update(meter, prefix='train')

            log_info('Evaluate on dev dataset ...')
            with torch.no_grad():
                meter = AverageMeter()
                for batch in dataset.iter_dev_dataset():
                    self.run_batch(model, None, batch, meter, False, device)

                self.log_stats('result', meter, prefix='dev')

            try:
                self.try_save(meter.accuracy)
            except Exception as err:
                log_warning('Can not save model: %s', err)

    def run_batch(self, model, optimizer, batch, meter, training, device):
        obs = to_device(batch[0], device)
        action_targets = batch[1].to(device)
        tile_targets = batch[2].to(device)

        if action_targets.size(0) <= 1:
            return

        if not training:
            model.eval()
            action_logits, tile_logits = model(**obs)
        else:
            model.train()
            optimizer.zero_grad()

            action_logits, tile_logits = model(**obs)
            action_loss = F.cross_entropy(action_logits, action_targets)
            tile_loss = F.cross_entropy(tile_logits, tile_targets)

            loss = action_loss + tile_loss

            loss.backward()
            optimizer.step()

            meter.add('loss', loss.item())
            meter.add('action_loss', action_loss.item())
            meter.add('tile_loss', tile_loss.item())

        add_stats(meter,
                  ['action', 'tile'],
                  [action_logits, tile_logits],
                  [action_targets, tile_targets])
