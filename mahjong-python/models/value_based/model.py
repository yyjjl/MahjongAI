# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..baseline_cnn.model import ModelOptions, BasicBlock, conv1x1, conv3x3


class ValueBasedModel(nn.Module):
    def __init__(self, options: ModelOptions):
        super().__init__()

        hidden_size = options.hidden_size
        count_dim = options.count_dim

        self.pack_offer_embedding = nn.Embedding(4, options.pack_offer_dim)
        self.pack_type_embedding = nn.Embedding(4, options.pack_type_dim)
        self.count_embedding = nn.Embedding(5, count_dim)

        # winds, pool, counts(3), packs(16)
        input_dim = 2 + 40 + count_dim * 3 + \
            (options.pack_offer_dim + options.pack_type_dim) * 16

        inplanes = options.initial_channels
        conv_layers = [
            conv3x3(input_dim, inplanes),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        ]
        for i in range(4):
            conv_layers.append(BasicBlock(inplanes, inplanes))
            conv_layers.append(BasicBlock(inplanes, inplanes))
            if i != 3:
                conv_layers.append(conv1x1(inplanes, inplanes * 2))
                conv_layers.append(nn.BatchNorm2d(inplanes * 2))
                inplanes *= 2

        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.conv_layers = nn.Sequential(*conv_layers)

        self.output_layers = nn.Sequential(
            nn.Linear(inplanes, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, counts, winds, pool, packs, **_):
        batch_size = pool.size(0)

        counts = self.count_embedding(counts).view(batch_size, 4, 9, -1).permute(0, 3, 1, 2)

        # [batch_size, 4, 9, 16, pack_offer_dim + pack_type_dim]
        packs = torch.cat([self.pack_offer_embedding(packs[..., 0]),
                           self.pack_type_embedding(packs[..., 1])], dim=-1)
        packs = packs.view(batch_size, 4, 9, -1).permute(0, 3, 1, 2)

        features = torch.cat([winds, pool, counts, packs], dim=1)
        features = self.conv_layers(features).view(batch_size, -1)

        return self.output_layers(features).squeeze(-1)
