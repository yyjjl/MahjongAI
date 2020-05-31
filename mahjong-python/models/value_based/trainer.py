# -*- coding: utf-8 -*-

from dataclasses import dataclass

import framework.session_manager as manager
from framework.utils import Options

from .dataloader import DataSetOptions
from .model import ValueBasedModel, ModelOptions

from ..utils import to_device
from ..baseline_cnn.trainer import SessionManager as SessionManagerBase


@dataclass
class MainOptions(manager.ManagerOptions, ModelOptions, DataSetOptions, Options):
    learning_rate: float
    weight_decay: float
    num_epoch_steps: int
    num_epochs: int


class SessionManager(SessionManagerBase):
    def build_model(self):
        return ValueBasedModel(self.options)

    def run_batch(self, model, optimizer, batch, meter, training, device):
        obs = to_device(batch[0], device)
        target_value = batch[1].to(device).float()

        if target_value.size(0) <= 1:
            return

        if not training:
            model.eval()
        else:
            model.train()
            optimizer.zero_grad()

        value = model(**obs)
        loss = ((target_value - value) ** 2).mean()

        if training:
            loss.backward()
            optimizer.step()

        meter.add('loss', loss.item())
        meter.add('accuracy', -loss.item())
