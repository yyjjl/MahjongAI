# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from dataclasses import dataclass

from .dataloader import ACTIONS1, ACTIONS2

from ..baseline_cnn.model import conv1x1, conv3x3, BasicBlock


@dataclass
class ModelOptions:
    pack_type_dim: int
    pack_offer_dim: int

    count_dim: int

    hidden_size: int
    initial_channels: int


class CNNModel(nn.Module):

    def __init__(self, options: ModelOptions):
        super().__init__()

        hidden_size = options.hidden_size
        count_dim = options.count_dim

        self.pack_offer_embedding = nn.Embedding(4, options.pack_offer_dim)
        self.pack_type_embedding = nn.Embedding(4, options.pack_type_dim)
        self.count_embedding = nn.Embedding(5, count_dim)

        # winds, pool, win_tile, counts(3), packs(16)
        input_dim = 2 + 40 + 1 + count_dim * 3 + \
            (options.pack_offer_dim + options.pack_type_dim) * 16

        self.conv_layers, inplanes = self.build_conv_layers(input_dim, options)

        self.action_hidden1 = nn.Sequential(
            nn.Linear(inplanes, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.action_hidden2 = nn.Sequential(
            nn.Linear(inplanes, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )

        self.output_layers1 = nn.ModuleList([
            nn.Linear(hidden_size, len(ACTIONS1)),
            nn.Linear(hidden_size, 35)
        ])
        self.output_layers2 = nn.ModuleList([
            nn.Linear(hidden_size, len(ACTIONS2)),
            nn.Linear(hidden_size, 35)
        ])
        self.mask_names = ['actions_mask', 'tiles_mask']

    def build_conv_layers(self, input_dim, options):
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

        return nn.Sequential(*conv_layers), inplanes

    def compute_features(self, counts, winds, pool, packs, win_tile):
        batch_size = pool.size(0)
        device = pool.device

        action_type = torch.zeros(batch_size, 36, device=device)
        if win_tile is None:
            action_type[:, -1] = 1
            output_layers = self.output_layers1
            action_hidden = self.action_hidden1
        else:
            output_layers = self.output_layers2
            action_hidden = self.action_hidden2

            assert (win_tile == 0).sum() == 0
            action_type[torch.arange(batch_size, device=device), win_tile - 1] = 1
        action_type = action_type.view(batch_size, 1, 4, 9)

        counts = self.count_embedding(counts).view(batch_size, 4, 9, -1).permute(0, 3, 1, 2)

        # [batch_size, 4, 9, 16, pack_offer_dim + pack_type_dim]
        packs = torch.cat([self.pack_offer_embedding(packs[..., 0]),
                           self.pack_type_embedding(packs[..., 1])], dim=-1)
        packs = packs.view(batch_size, 4, 9, -1).permute(0, 3, 1, 2)

        features = torch.cat([winds, pool, action_type, counts, packs], dim=1)
        features = self.conv_layers(features).view(batch_size, -1)
        features = action_hidden(features)

        return features, output_layers

    def compute_logits(self, features, output_layers, masks):
        result = []
        for name, layer in zip(self.mask_names, output_layers):
            logits = layer(features)
            logits = torch.where(masks[name], logits, torch.zeros_like(logits) - 10000)
            result.append(logits)
        return result

    def forward(self, counts, winds, pool, packs, win_tile=None, **masks):
        features, output_layers = self.compute_features(counts, winds, pool, packs, win_tile)

        return self.compute_logits(features, output_layers, masks)
