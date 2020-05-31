# -*- coding: utf-8 -*-

import multiprocessing as mp
import pickle
from glob import glob

import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

import PyMahjong as mj
from framework.logging import log_error, open_file
from framework.utils import ProgressReporter
from IO.replay_reader import SPEICAL_FAN_NAMES, Action

MT = mj.MessageType
IGNORE_INDEX = -100
ACTIONS = [MT.kPlay,
           MT.kKong,  # 暗杠
           MT.kExtraKong,
           MT.kPass,
           MT.kPung,
           MT.kKong,  # 明杠
           MT.kChow, MT.kChow, MT.kChow]

PASS_INDEX = ACTIONS.index(MT.kPass)
PLAY_INDEX = ACTIONS.index(MT.kPlay)
HIDDEN_KONG_INDEX = ACTIONS.index(MT.kKong)
ADD_KONG_INDEX = ACTIONS.index(MT.kExtraKong)
PUNG_INDEX = ACTIONS.index(MT.kPung)
KONG_INDEX = ACTIONS.index(MT.kKong, HIDDEN_KONG_INDEX + 1)
CHOW_INDEX = ACTIONS.index(MT.kChow)


@dataclass
class DataSetOptions:
    only_winner_data: bool
    exclude_special_fans: bool
    batch_size: int


def tile_to_map(tile):
    assert tile != 0, '??? tile != 0'
    tile_map = np.zeros((36, ), dtype=np.float32)
    tile_map[tile - 1] = 1
    return tile_map.reshape((1, 4, 9))


def compute_mask1(game, player_id, win_tile):
    actions_mask = np.zeros((len(ACTIONS), ), dtype=np.uint8)

    actions_mask[PLAY_INDEX] = 1
    if game.player(player_id).count(win_tile, ignore_packs=True) == 4:  # 暗杠
        actions_mask[HIDDEN_KONG_INDEX] = 1
    if game.can_add_kong(player_id, win_tile):   # 补杠
        actions_mask[ADD_KONG_INDEX] = 1
    return actions_mask


def compute_mask2(player, last_player_id, win_tile):
    count = player.count(win_tile, True)
    nearby_counts = player.nearby_counts(win_tile)
    actions_mask = np.zeros((len(ACTIONS), ), dtype=np.uint8)

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


class Loader:
    def __init__(self, options: DataSetOptions=None):
        self.options = options

    def obs_of_player(self, game, player_id):
        player = game.player(player_id)

        counts = np.stack(game.view_of_player(player_id), axis=-1)
        pool = game.tiles_pool(player_id, 10).reshape((-1,))
        packs = game.all_packs(player_id)

        winds = np.zeros((2, 4, 9), dtype=np.float32)
        winds[0, 3, int(player.prevalent_wind)] = 1
        winds[1, 3, int(player.seat_wind)] = 1

        new_pool = np.zeros((40, 36), dtype=np.float32)
        new_pool[np.arange(40), np.maximum(0, pool - 1)] = 1
        new_pool[pool == 0] = 0
        pool = new_pool.reshape((40, 4, 9))

        pack_offer_and_type = packs[..., :2].reshape((-1, 2))
        pack_tile = packs[..., 2].reshape((-1,))

        new_packs = np.zeros((36, 16, 2), dtype=np.int64)
        new_packs[np.where(pack_tile != 0, pack_tile - 1, 34),
                  np.arange(16), :] = pack_offer_and_type
        packs = new_packs.reshape(4, 9, 16, 2)

        return {
            'tiles_mask': player.tiles_mask().astype(np.uint8),
            'winds': winds,
            'counts': counts,
            'packs': packs,
            'pool': pool
        }

    def init(self, game, replay):
        master_id = None
        for player_id, (tiles, flower_count) in enumerate(replay['initial_tiles']):
            if len(tiles) == 14:
                assert master_id is None
                master_id = player_id
                tile = tiles.pop()
                game.draw(player_id, tile)
                last_action = Action(player_id=master_id, type=MT.kDraw, info_tile=tile)

            assert game.init_tiles(player_id, tiles, flower_count), '??? 初始化失败'

        game.init_seat(replay['wind'], master_id)

        return last_action

    def after_draw(self, game, current_action):
        current_player_id = current_action.player_id
        obs = self.obs_of_player(game, current_player_id)
        action_index = ACTIONS.index(current_action.type)
        output_tile = current_action.output_tile or IGNORE_INDEX

        player = game.player(current_player_id)
        win_tile = player.win_tile_index

        actions_mask = compute_mask1(game, current_player_id, win_tile)
        obs['actions_mask'] = actions_mask
        if actions_mask[action_index] == 0:
            # 由于人类数据, 任何时候可以补杠/暗杠, 所以 actions_mask 有问题
            action_index = IGNORE_INDEX

        obs['win_tile'] = tile_to_map(win_tile)
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
        obs['win_tile'] = tile_to_map(win_tile)

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
            action_index = ACTIONS.index(tp, PASS_INDEX)

        return obs, action_index, output_tile

    def simulate(self, replay, return_game):
        options = self.options
        only_winner_data = options.only_winner_data

        game = mj.Game()
        actions = replay['actions']

        if only_winner_data and actions[-1].type != MT.kWin:
            return None

        if options.exclude_special_fans and SPEICAL_FAN_NAMES.intersection(replay['fan_names']):
            return None  # 有特殊牌型

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
                game.win(action.player_id, action.info_tile)  # 检查是否真的和了
                break

            if step > 0:
                last_action = actions[step - 1]

            if last_action.player_id == current_player_id and last_action.type == MT.kDraw:
                if (not only_winner_data) or win_player_id == current_player_id:
                    examples.append(self.after_draw(game, action))

                # 上一次自己摸牌, 可以打, 补杠, 暗杠
                if request_type == MT.kPlay:
                    assert game.play(current_player_id, output_tile), '打牌失败'
                elif request_type == MT.kKong:
                    assert info_tile is not None, '暗杠需要指定牌'
                    assert game.kong(current_player_id, info_tile, is_hidden=True), '暗杠失败'
                elif request_type == MT.kExtraKong:
                    assert game.add_kong(current_player_id, info_tile), '补杠失败'
                continue

            if only_winner_data:
                player_ids = [win_player_id]
            else:
                player_ids = [0, 1, 2, 3]
            for selected_player_id in player_ids:
                example = self.after_play(game, action, last_action, selected_player_id)
                if example is not None:
                    examples.append(example)

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


class DataSet:
    LoaderClass = Loader

    def __init__(self, options, patterns):
        self.batch_size = options.batch_size
        self.loader = self.LoaderClass(options)

        files = []
        for pattern in patterns:
            files.extend(glob(pattern))

        sep = int(0.95 * len(files))
        self.train_files = files[:sep]
        self.dev_files = files[sep:]

    def iter_train_dataset(self):
        yield from self._iter_dataset(self.train_files)

    def iter_dev_dataset(self):
        yield from self._iter_dataset(self.dev_files)

    def _iter_dataset(self, files):
        for file in files:
            yield from DataLoader(load_replays(self.loader, file),
                                  batch_size=self.batch_size, shuffle=True)
