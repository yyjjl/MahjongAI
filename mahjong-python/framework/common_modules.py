# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, attn_size, input_dim):
        super().__init__()
        self.attn_size = attn_size

        # Attention ###########################################################
        # $$ e^t_i = v^T \tanh(W_h h_i + b_{attn}) $$
        self.attn_v = nn.Linear(self.attn_size, 1, bias=False)
        self.attn_W_h = nn.Linear(input_dim, self.attn_size, bias=True)

    def forward(self, input_features, input_mask=None):
        """
        input_state: [*, num_keys, input_dim]
        input_mask: [*, num_keys, 1]
        """
        # shape: [*, num_keys, 1]
        attn_e = self.attn_v(torch.tanh(self.attn_W_h(input_features)))
        # shape: [*, num_keys, 1]
        attn_dist = F.softmax(attn_e, dim=-2)
        if input_mask is not None:
            attn_dist = attn_dist * input_mask
        # shape: [*, 1, 1]
        masked_sums = attn_dist.sum(dim=-2, keepdim=True) + 1e-13
        attn_dist = attn_dist / masked_sums  # re-normalize

        return attn_dist


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super().__init__()
        self.nonlinears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, input):
        """
        :param input: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        """
        for gate, linear, nonlinear in zip(self.gates, self.linears,
                                           self.nonlinears):
            gate = torch.sigmoid(gate(input))
            nonlinear = self.f(nonlinear(input))
            linear = linear(input)

            input = gate * nonlinear + (1 - gate) * linear
        return input
