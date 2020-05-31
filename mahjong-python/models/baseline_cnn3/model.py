# -*- coding: utf-8 -*-

import math

import torch.nn as nn
import torch.nn.functional as F

from ..baseline_cnn2.model import conv3x3, conv1x1
from ..baseline_cnn2.model import CNNModel as CNNModelBase


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        layers = [
            conv1x1(inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            conv3x3(planes, planes, stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            conv1x1(planes, planes * 4),
            nn.BatchNorm2d(planes * 4)
        ]

        self.layers = nn.Sequential(*layers)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_dim, initial_channels):
        super().__init__()

        self.inplanes = initial_channels
        self.conv_layers = nn.Sequential(
            conv3x3(input_dim, initial_channels),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            self._make_layer(block, initial_channels, layers[0]),
            self._make_layer(block, initial_channels * 2, layers[1]),
            self._make_layer(block, initial_channels * 4, layers[2]),
            self._make_layer(block, initial_channels * 8, layers[3]),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x).view(x.size(0), -1)


class CNNModel(CNNModelBase):

    def build_conv_layers(self, input_dim, options):
        inplanes = options.initial_channels
        conv_layers = ResNet(Bottleneck, [3, 4, 6, 3], input_dim, inplanes)
        inplanes = inplanes * 8 * Bottleneck.expansion
        return conv_layers, inplanes
