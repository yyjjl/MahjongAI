# -*- coding: utf-8 -*-

import math
import multiprocessing as mp
import pickle
from typing import Union

import numpy as np
from dataclasses import dataclass

import PyMahjong as mj
from framework.logging import log_error, open_file
from framework.utils import ProgressReporter

from ..baseline_cnn.dataloader import DataSet as DataSetBase
from ..baseline_cnn.dataloader import Loader as LoaderBase

MT = mj.MessageType


@dataclass
class DataSetOptions:
    batch_size: int
    use_weighted_value: bool
    value_type: Union[str, float]


class Loader(LoaderBase):
    def try_win(self, game, win_player_id, examples):
        winner = game.player(win_player_id)
        # 去掉花牌
        value = game.try_win(win_player_id, apply_change=True) - winner.flower_count
        assert value >= 8

        value_type = self.options.value_type
        if not self.options.use_weighted_value:
            value = 16

        final_obs = self.obs_of_player(game, win_player_id)
        counts = final_obs['counts']
        target_hand_count = counts[..., 1].ravel()
        target_all_count = counts[..., 2].ravel()
        target_pack_count = target_all_count - target_hand_count
        assert all(target_pack_count >= 0)
        assert ((counts < 0) | (counts > 4)).sum() == 0

        new_examples = [(final_obs, value)]
        for i, obs in enumerate(examples):
            counts = obs['counts']
            hand_count = counts[..., 1].ravel()
            all_count = counts[..., 2].ravel()
            assert all((target_pack_count - (all_count - hand_count)) >= 0)

            diff = target_all_count - all_count
            if value_type == 'probs':
                scale = 0
                rest_count = counts[..., 0].ravel()
                rest_total_count = rest_count.sum()
                for tile, count in enumerate(diff):
                    if count == 0:
                        continue
                    if count > 0:  # 少
                        assert rest_count[tile] >= count
                        for i in range(count):
                            scale += math.log((rest_count[tile] - i) / rest_total_count)
                            rest_total_count -= 1
                new_value = scale + math.log(value)
            elif value_type == 'linear':
                scale = 1 - np.maximum(diff, 0).sum() / 7.0
                new_value = scale * value
            elif isinstance(value_type, (int, float)):
                scale = value_type ** (len(examples) - i)
                new_value = scale * value
            else:
                new_value = value

            new_examples.append((obs, new_value))

        return new_examples

    def simulate(self, replay, return_game=False):
        game = mj.Game()
        actions = replay['actions']

        if actions[-1].type != MT.kWin:
            return None

        last_action = self.init(game, replay)

        win_player_id = actions[-1].player_id
        examples = []
        for step, action in enumerate(actions):
            request_type = action.type
            current_player_id = action.player_id
            info_tile = action.info_tile
            output_tile = action.output_tile

            if request_type == MT.kExtraFlower:  # 补花由系统控制
                game.add_flower(current_player_id)
                continue

            if request_type == MT.kWin:
                assert action.player_id == win_player_id
                examples = self.try_win(game, win_player_id, examples)  # 检查是否真的和了
                break

            if step > 0:
                last_action = actions[step - 1]

            if last_action.player_id == current_player_id and last_action.type == MT.kDraw:
                # 上一次自己摸牌, 可以打, 补杠, 暗杠
                if request_type == MT.kPlay:
                    assert game.play(current_player_id, output_tile), '打牌失败'
                elif request_type == MT.kKong:
                    assert info_tile is not None, '暗杠需要指定牌'
                    assert game.kong(current_player_id, info_tile, is_hidden=True), '暗杠失败'
                elif request_type == MT.kExtraKong:
                    assert game.add_kong(current_player_id, info_tile), '补杠失败'

                if win_player_id == current_player_id:
                    examples.append(self.obs_of_player(game, win_player_id))
                continue

            # 上一次别人打出牌, 那么可以碰, 吃, 杠或过
            if request_type == MT.kDraw:  # 摸牌由系统控制
                game.draw(current_player_id, info_tile)
            elif request_type == MT.kChow:
                assert game.chow(current_player_id, info_tile, output_tile), '吃牌失败'
            elif request_type == MT.kPung:
                assert game.pung(current_player_id, output_tile), '碰牌失败'
            elif request_type == MT.kKong:  # 明杠
                assert info_tile is None
                assert game.kong(current_player_id), '明杠失败'
            else:
                raise Exception(f'非法操作: {action}')

            if request_type != MT.kDraw and win_player_id == current_player_id:
                examples.append(self.obs_of_player(game, win_player_id))

        if return_game:
            return game, examples

        return examples

    def worker(self, replay, return_game=False):
        try:
            return self.simulate(replay, return_game)
        except Exception as err:
            log_error(err, '%s', replay['file'])
            return None


def load_replays(loader, path, num_processes=mp.cpu_count()):
    replays = pickle.load(open_file(path, 'rb'))
    progress = ProgressReporter(len(replays), step=100)
    pool = mp.Pool(processes=num_processes)

    examples = []
    try:
        for result in progress(pool.imap_unordered(loader.worker, replays, chunksize=25)):
            if result is None:
                continue

            if isinstance(result, tuple):
                examples.append(result)
            else:
                examples.extend(result)
    finally:
        pool.terminate()

    return examples


class DataSet(DataSetBase):
    LoaderClass = Loader
