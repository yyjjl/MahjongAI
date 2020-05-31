# -*- coding: utf-8 -*-

import torch

from framework.logging import log_info
from framework.utils import AverageMeter

from ..baseline_cnn import SessionManager as SessionManagerBase
from .dataloader import is_type2
from .model import CNNModel


class SessionManager(SessionManagerBase):
    def build_model(self):
        return CNNModel(self.options)

    def train(self, dataset):
        model = self.model
        optimizer = self.optimizer
        device = self.device

        for i in range(self.num_epochs):
            meter1 = AverageMeter()
            meter2 = AverageMeter()
            for batch in dataset.iter_train_dataset():
                if is_type2(batch):
                    meter = meter2
                    prefix = 'train_type2'
                else:
                    meter = meter1
                    prefix = 'train_type1'
                self.run_batch(model, optimizer, batch, meter, True, device)
                self.on_update(meter, prefix=prefix)

            log_info('Evaluate on dev dataset ...')
            with torch.no_grad():
                meter1 = AverageMeter()
                meter2 = AverageMeter()
                for batch in dataset.iter_dev_dataset():
                    if is_type2(batch):
                        meter = meter2
                    else:
                        meter = meter1
                    self.run_batch(model, None, batch, meter, False, device)

                self.log_stats('result', meter1, prefix='dev_type1')
                self.log_stats('result', meter2, prefix='dev_type2')

            self.try_save(meter.accuracy)
