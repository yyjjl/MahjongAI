# -*- coding: utf-8 -*-

import numpy as np

import PyMahjong as mj
from IO.replay_reader import Action
from models.baseline_cnn.dataloader import ACTIONS, CHOW_INDEX, compute_mask1, compute_mask2

MT = mj.MessageType


class RandomAgent:
    def __init__(self, name=None):
        self.name = name or 'Random'
        self.reset()

    def reset(self):
        self.final_rewards = [0, 0, 0, 0]

    def finish(self, player_id, reward):
        self.final_rewards[player_id] = reward

    def add_reward(self, player_id, reward):
        pass

    def sample(self, weights):
        weights = weights.astype(np.float32) / weights.sum()
        return np.random.choice(np.arange(len(weights)), 1, p=weights)[0]

    def _after_draw(self, player_id, game):
        player = game.player(player_id)
        win_tile = player.win_tile_index

        action_index = self.sample(compute_mask1(game, player_id, win_tile))

        action = Action(player_id=player_id, type=ACTIONS[action_index], info_tile=win_tile)
        if action.type == MT.kPlay:
            action.output_tile = self.sample(player.tiles_mask())

        return action

    def _after_play(self, player_id, game, output_tile):
        player = game.player(player_id)
        last_message = game.history(game.turn_ID - 1)
        win_tile = last_message.output_tile_index
        last_player_id = last_message.player_id

        assert last_player_id != player_id

        actions_mask = compute_mask2(player, last_player_id, win_tile)
        if actions_mask.sum() == 1:
            return Action(player_id=player_id, type=mj.MessageType.kPass)

        action_index = self.sample(actions_mask)
        action = Action(player_id=player_id, type=ACTIONS[action_index])

        tiles_mask = player.tiles_mask()
        if action.type == MT.kChow:
            info_tile = action_index - CHOW_INDEX - 1 + win_tile
            for i in [-1, 0, 1]:
                tile = info_tile + i
                if player.count(tile, ignore_packs=True) == 1 and tile != win_tile:
                    tiles_mask[tile] = 0

            action.info_tile = info_tile
            action.output_tile = self.sample(tiles_mask)
        elif action.type == MT.kPung:
            tiles_mask[win_tile] = 0
            action.output_tile = self.sample(tiles_mask)

        return action

    def step(self, player_id, game, output_tile=None):
        if game.try_win(player_id) > 0:
            return Action(player_id=player_id, type=mj.MessageType.kWin)

        if output_tile is not None:
            return self._after_play(player_id, game, output_tile)
        return self._after_draw(player_id, game)
