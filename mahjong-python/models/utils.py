# -*- coding: utf-8 -*-

import torch


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return x.__class__(to_device(value, device) for value in x)
    if isinstance(x, dict):
        return {key: to_device(value, device) for key, value in x.items()}
    return x


def add_stats(meter, names, logits_list, targets_list, correct=None):
    if correct is None:
        correct = torch.ones(logits_list[0].size(0),
                             dtype=getattr(torch, 'bool', torch.uint8),
                             device=logits_list[0].device)

    total_count = correct.sum().item()
    for name, logits, targets in zip(names, logits_list, targets_list):
        pred = (logits.argmax(dim=-1) == targets)
        mask = targets != -100
        correct &= (~mask) | pred
        pred = pred[mask]
        meter.add(f'{name}_accuracy', pred.sum().item(), pred.size(0))

    meter.add('accuracy', correct.sum().item(), total_count)


def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = lengths.max()

    mask = torch.arange(0, maxlen, dtype=lengths.dtype, device=lengths.device)
    return mask < lengths.unsqueeze(-1)
