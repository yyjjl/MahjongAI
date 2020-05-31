# -*- coding: utf-8 -*-

import numpy as np

from ..baseline_cnn.dataloader import Loader as LoaderBase
from ..baseline_cnn.dataloader import DataSet as DataSetBase
from ..baseline_cnn.dataloader import IGNORE_INDEX


def pad_and_truncate(examples, max_length):
    length = len(examples)
    if length == 0:
        return None

    if length > max_length:
        examples = examples[:length]
    else:
        obs = examples[0][0]

        obs_padding = {}
        for name, value in obs.items():
            obs_padding[name] = np.zeros_like(value)

        padding = obs_padding, IGNORE_INDEX, IGNORE_INDEX
        while len(examples) < max_length:
            examples.append(padding)

    all_obs = {}
    for key in examples[0][0].keys():
        all_obs[key] = np.stack([example[0][key] for example in examples], axis=0)

    return (length,
            all_obs,
            *(np.stack([example[i] for example in examples], axis=0) for i in range(1, 3)))


class Loader(LoaderBase):
    def __init__(self, options):
        assert options.only_winner_data

        super().__init__(options)

    def simulate(self, replay, return_game=False):
        examples = super().simulate(replay, return_game)
        if examples is not None:
            return pad_and_truncate(examples, 40)


class DataSet(DataSetBase):
    LoaderClass = Loader
