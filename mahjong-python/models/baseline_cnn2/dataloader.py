# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

import PyMahjong as mj

from ..baseline_cnn.dataloader import IGNORE_INDEX
from ..baseline_cnn.dataloader import DataSet as DataSetBase
from ..baseline_cnn.dataloader import Loader as LoaderBase
from ..baseline_cnn.dataloader import load_replays

MT = mj.MessageType

ACTIONS1 = [MT.kPlay, MT.kKong, MT.kExtraKong]
ACTIONS2 = [MT.kPass, MT.kPung, MT.kKong, MT.kChow, MT.kChow, MT.kChow]

PLAY_INDEX = ACTIONS1.index(MT.kPlay)
HIDDEN_KONG_INDEX = ACTIONS1.index(MT.kKong)
ADD_KONG_INDEX = ACTIONS1.index(MT.kExtraKong)

PASS_INDEX = ACTIONS2.index(MT.kPass)
PUNG_INDEX = ACTIONS2.index(MT.kPung)
KONG_INDEX = ACTIONS2.index(MT.kKong)
CHOW_INDEX = ACTIONS2.index(MT.kChow)


@dataclass
class DataSetOptions:
    only_winner_data: bool
    exclude_special_fans: bool
    batch_size: int


def is_type2(x):
    return 'win_tile' in x[0]


def compute_mask1(game, player_id, win_tile):
    actions_mask = np.zeros((len(ACTIONS1), ), dtype=np.uint8)

    actions_mask[PLAY_INDEX] = 1
    if game.player(player_id).count(win_tile, ignore_packs=True) == 4:  # 暗杠
        actions_mask[HIDDEN_KONG_INDEX] = 1
    if game.can_add_kong(player_id, win_tile):   # 补杠
        actions_mask[ADD_KONG_INDEX] = 1
    return actions_mask


def compute_mask2(player, last_player_id, win_tile):
    count = player.count(win_tile, True)
    nearby_counts = player.nearby_counts(win_tile)
    actions_mask = np.zeros((len(ACTIONS2), ), dtype=np.uint8)

    actions_mask[PASS_INDEX] = 1  # Pass
    if player.is_upstream(last_player_id) and win_tile <= 27:  # 数牌
        for i in range(3):
            if nearby_counts[i] >= 1 and nearby_counts[i + 1] >= 1:
                actions_mask[CHOW_INDEX + i] = 1
    if count >= 2:
        actions_mask[PUNG_INDEX] = 1
    if count == 3:
        actions_mask[KONG_INDEX] = 1

    return actions_mask


class Loader(LoaderBase):
    def after_draw(self, game, current_action):
        current_player_id = current_action.player_id
        obs = self.obs_of_player(game, current_player_id)
        action_index = ACTIONS1.index(current_action.type)
        output_tile = current_action.output_tile or IGNORE_INDEX

        player = game.player(current_player_id)
        win_tile = player.win_tile_index

        actions_mask = compute_mask1(game, current_player_id, win_tile)
        obs['actions_mask'] = actions_mask
        if actions_mask[action_index] == 0:
            # 由于人类数据, 任何时候可以补杠/暗杠, 所以 actions_mask 有问题
            action_index = IGNORE_INDEX
        return obs, action_index, output_tile

    def after_play(self, game, current_action, last_action, selected_player_id):
        win_tile = last_action.output_tile
        if win_tile is None:
            return None

        last_player_id = last_action.player_id
        if last_player_id == selected_player_id:
            return None  # 上一次是自己打出牌, 跳过

        action_player_id = current_action.player_id
        request_type = current_action.type
        info_tile = current_action.info_tile

        player = game.player(selected_player_id)  # 以 selected_player_id 玩家的视角
        actions_mask = compute_mask2(player, last_player_id, win_tile)

        tp = request_type
        if action_player_id != selected_player_id:  # 不是当前动作执行者
            if tp == MT.kKong:
                # 由于杠优先级高, 所以不知道是玩家主动不碰/吃, 还是碰/吃了没成
                # 功, 不用这组数据
                actions_mask[CHOW_INDEX:] = 0
                actions_mask[PUNG_INDEX] = 0
            elif tp == MT.kPung:
                # 由于碰优先级高, 所以不知道是玩家主动不吃, 还是吃了没成功, 不
                # 用这组数据
                actions_mask[CHOW_INDEX:] = 0
            # 由于不是执行者, 所以没有打出的牌
            tp = MT.kPass
            output_tile = IGNORE_INDEX
        else:
            output_tile = current_action.output_tile or IGNORE_INDEX

        if tp == MT.kDraw:
            tp = MT.kPass

        if actions_mask.sum() == 1:  # 不用这组数据
            return None

        obs = self.obs_of_player(game, selected_player_id)
        obs['actions_mask'] = actions_mask
        obs['win_tile'] = win_tile

        tiles_mask = obs['tiles_mask']
        if tp == MT.kChow and output_tile != IGNORE_INDEX:
            for i in [-1, 0, 1]:
                tile = info_tile + i
                # 考虑吃了又打出 output_tile == win_tile
                if player.count(tile, ignore_packs=True) == 1 and output_tile != win_tile:
                    tiles_mask[tile] = 0
            assert tiles_mask[output_tile], '吃牌错误'
            action_index = CHOW_INDEX + info_tile - win_tile + 1
        else:
            action_index = ACTIONS2.index(tp)

        return obs, action_index, output_tile


class DataSet(DataSetBase):
    LoaderClass = Loader

    def _iter_dataset(self, files):
        for file in files:
            examples = load_replays(self.loader, file)
            examples1 = []
            examples2 = []
            for example in examples:
                if is_type2(example):
                    examples2.append(example)
                else:
                    examples1.append(example)

            del examples
            yield from DataLoader(examples1, batch_size=self.batch_size, shuffle=True)
            yield from DataLoader(examples2, batch_size=self.batch_size, shuffle=True)
