# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from .dataloader import ACTIONS


@dataclass
class ModelOptions:
    pack_type_dim: int
    pack_offer_dim: int

    count_dim: int

    hidden_size: int
    initial_channels: int


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity, inplace=True)

        return out


class CNNModel(nn.Module):
    ACTIONS = ACTIONS

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

        self.hidden_layers = nn.Sequential(
            nn.Linear(inplanes, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size, len(ACTIONS)),
            nn.Linear(hidden_size, 35)
        ])

        self.mask_names = ['actions_mask', 'tiles_mask']

    def compute_features(self, counts, winds, pool, packs, win_tile):
        batch_size = pool.size(0)

        counts = self.count_embedding(counts).view(batch_size, 4, 9, -1).permute(0, 3, 1, 2)

        # [batch_size, 4, 9, 16, pack_offer_dim + pack_type_dim]
        packs = torch.cat([self.pack_offer_embedding(packs[..., 0]),
                           self.pack_type_embedding(packs[..., 1])], dim=-1)
        packs = packs.view(batch_size, 4, 9, -1).permute(0, 3, 1, 2)

        features = torch.cat([winds, pool, win_tile, counts, packs], dim=1)
        features = self.conv_layers(features).view(batch_size, -1)
        features = self.hidden_layers(features)

        return features

    def compute_logits(self, features, masks):
        result = []
        for name, layer in zip(self.mask_names, self.output_layers):
            logits = layer(features)
            logits = torch.where(masks[name], logits, torch.zeros_like(logits) - 10000)
            result.append(logits)
        return result

    def forward(self, counts, winds, pool, packs, win_tile, **masks):
        features = self.compute_features(counts, winds, pool, packs, win_tile)

        return self.compute_logits(features, masks)
