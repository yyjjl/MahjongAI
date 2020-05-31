# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..baseline_cnn.model import ModelOptions, CNNModel


class CNNLSTMModel(CNNModel):
    def __init__(self, options: ModelOptions):
        super().__init__(options)

        hidden_size = options.hidden_size

        inplanes = self.hidden_layers[0].in_features
        self.hidden_layers = nn.Sequential(
            nn.Linear(inplanes, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
        )
        self.features_lstm = nn.LSTM(hidden_size // 2, hidden_size // 2, num_layers=2,
                                     batch_first=True)

    def forward(self, counts, winds, pool, packs, win_tile, **masks):
        args = counts, winds, pool, packs, win_tile

        batch_size, seq_length = args[0].shape[:2]
        args = [((arg.view(-1, *arg.shape[2:])) if arg is not None else arg) for arg in args]

        features = self.compute_features(*args)

        features = features.view(batch_size, seq_length, -1)
        lstm_output, _ = self.features_lstm(features)
        features = torch.cat([features, lstm_output], dim=-1)

        return self.compute_logits(features, masks)
