# -*- coding: utf-8 -*-

import torch.nn.functional as F

from ..utils import add_stats, to_device, sequence_mask
from .model import CNNLSTMModel

from ..baseline_cnn.trainer import MainOptions, SessionManager as SessionManagerBase


class SessionManager(SessionManagerBase):
    def __init__(self, options: MainOptions, training=True):
        super().__init__(options, training)

    def build_model(self):
        return CNNLSTMModel(self.options)

    def run_batch(self, model, optimizer, batch, meter, training, device):
        obs = to_device(batch[1], device)
        action_targets = batch[2].to(device)
        tile_targets = batch[3].to(device)

        seq_length = action_targets.size(1)

        tile_targets = tile_targets.view(-1)
        action_targets = action_targets.view(-1)

        if not training:
            model.eval()
        else:
            model.train()
            optimizer.zero_grad()

        action_logits, tile_logits = model(**obs)
        action_logits = action_logits.view(-1, action_logits.size(-1))
        tile_logits = tile_logits.view(-1, tile_logits.size(-1))
        if training:
            action_loss = F.cross_entropy(action_logits, action_targets)
            tile_loss = F.cross_entropy(tile_logits, tile_targets)

            loss = action_loss + tile_loss

            loss.backward()
            optimizer.step()

            meter.add('loss', loss.item())
            meter.add('action_loss', action_loss.item())
            meter.add('tile_loss', tile_loss.item())

        length = batch[0].to(device)
        correct = sequence_mask(length, seq_length).view(-1)
        add_stats(meter,
                  ['action', 'tile'],
                  [action_logits, tile_logits],
                  [action_targets, tile_targets],
                  correct=correct)
