# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from dataclasses import dataclass

import PyMahjong as mj
from IO.replay_reader import Action
from models.baseline_cnn.dataloader import (ACTIONS,
                                            CHOW_INDEX,
                                            IGNORE_INDEX,
                                            compute_mask1,
                                            compute_mask2,
                                            tile_to_map)

INF = float('inf')
MT = mj.MessageType


@dataclass
class Example:
    observation: tuple
    reward: float = None
    gt: float = 0

    def to_tuple(self):
        return self.observation, self.reward, self.gt


def to_batch_tensors(obs, device):
    return {key: (torch.from_numpy(value)
                  if not torch.is_tensor(value) else value).unsqueeze(0).to(device)
            for key, value in obs.items()}


class Agent:
    def __init__(self, model, device, discount, obs_of_player_fn, name=None):
        self.name = name or 'Anonymous'

        self.obs_of_player = obs_of_player_fn

        self.model = model
        self.device = device

        self.discount = discount

        self.reset()

    def reset(self):
        self.examples = [[], [], [], []]
        self.final_rewards = [0, 0, 0, 0]

    def finish(self, player_id, reward):
        self.final_rewards[player_id] = reward

        discount = self.discount
        examples = self.examples[player_id]

        for example in reversed(examples):  # type: Example
            reward = example.gt = example.reward + discount * reward

    def sample(self, game, logits):
        # if game.rest_wall_count > 70:
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        # return logits.argmax(-1).item()

    def _after_draw(self, player_id, game):
        examples = self.examples[player_id]

        player = game.player(player_id)
        win_tile = player.win_tile_index

        actions_mask = torch.from_numpy(compute_mask1(game, player_id, win_tile))

        obs = self.obs_of_player(game, player_id)
        obs['actions_mask'] = actions_mask
        obs['win_tile'] = tile_to_map(win_tile)

        obs = to_batch_tensors(obs, self.device)
        with torch.no_grad():
            self.model.eval()
            action_logits, tile_logits = self.model(**obs)

        action_index = self.sample(game, action_logits)
        action = Action(player_id=player_id, type=ACTIONS[action_index], info_tile=win_tile)
        if action.type == MT.kPlay:
            action.output_tile = self.sample(game, tile_logits)

        output_tile = action.output_tile or IGNORE_INDEX
        examples.append(Example(observation=(obs, action_index, output_tile)))

        return action

    def _after_play(self, player_id, game, output_tile):
        examples = self.examples[player_id]

        player = game.player(player_id)
        last_message = game.history(game.turn_ID - 1)
        win_tile = last_message.output_tile_index
        last_player_id = last_message.player_id

        assert last_player_id != player_id

        actions_mask = compute_mask2(player, last_player_id, win_tile)
        if actions_mask.sum() == 1:
            examples.append(None)
            return Action(player_id=player_id, type=mj.MessageType.kPass)

        obs = self.obs_of_player(game, player_id)
        obs['actions_mask'] = actions_mask
        obs['win_tile'] = tile_to_map(win_tile)

        obs = to_batch_tensors(obs, self.device)
        with torch.no_grad():
            self.model.eval()
            action_logits, tile_logits = self.model(**obs)

        action_index = self.sample(game, action_logits)
        action = Action(player_id=player_id, type=ACTIONS[action_index])

        if action.type == MT.kChow:
            info_tile = action_index - CHOW_INDEX - 1 + win_tile
            for i in [-1, 0, 1]:
                tile = info_tile + i
                if player.count(tile, ignore_packs=True) == 1 and tile != win_tile:
                    tile_logits[0, tile] = -INF

            action.info_tile = info_tile
            action.output_tile = self.sample(game, tile_logits)
        elif action.type == MT.kPung:
            tile_logits[0, win_tile] = -INF
            action.output_tile = self.sample(game, tile_logits)

        output_tile = action.output_tile or IGNORE_INDEX
        examples.append(Example(observation=(obs, action_index, output_tile)))

        return action

    def add_reward(self, player_id, reward):
        examples = self.examples[player_id]
        example = examples[-1]
        if example is not None:
            assert example.reward is None
            # example.reward = reward
            example.reward = reward
        else:
            examples.pop()

    def step(self, player_id, game, output_tile=None):
        if game.try_win(player_id) > 0:
            self.examples[player_id].append(None)
            return Action(player_id=player_id, type=mj.MessageType.kWin)

        if output_tile is not None:
            return self._after_play(player_id, game, output_tile)
        return self._after_draw(player_id, game)
