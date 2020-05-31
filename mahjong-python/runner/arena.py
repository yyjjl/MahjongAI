# -*- coding: utf-8 -*-

import PyMahjong as mj
import random

from IO.replay_reader import Action
from RL.arena import ALL_TILES, PRIOIRITY, is_flower


class Arena:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def init_game(self, tiles_wall=None, prevalent_wind=None):
        game = mj.Game()

        if prevalent_wind is None:
            prevalent_wind = random.randrange(4)

        master_id = 0
        game.init_seat(mj.Wind(prevalent_wind), master_id)

        if tiles_wall is None:
            tiles_wall = ALL_TILES.copy()
            random.shuffle(tiles_wall)

        flower_counts = [0, 0, 0, 0]
        initial_tiles = [[], [], [], []]
        for _ in range(13):
            for i in range(4):
                while True:
                    tile = tiles_wall.pop()
                    if is_flower(tile):
                        flower_counts[i] += 1
                    else:
                        initial_tiles[i].append(tile)
                        break

        for i, agent in enumerate(self.agents):
            assert game.init_tiles(i, initial_tiles[i], flower_counts[i]), '初始化失败'

            agent.init_seat(mj.Wind(prevalent_wind), i)
            agent.init_tiles(initial_tiles[i], flower_counts[i])

        return game, master_id, tiles_wall

    def _execute_draw(self, game, current_player_id, tile):
        game.draw(current_player_id, tile)

        agent = self.agents[current_player_id]
        action = agent.step(current_player_id, game)  # type: Action

        action_type = action.type
        output_tile = action.output_tile
        info_tile = action.info_tile

        if action_type == mj.MessageType.kWin:
            pass
        elif action_type == mj.MessageType.kPlay:
            assert game.play(current_player_id, output_tile), '打牌失败'
        elif action_type == mj.MessageType.kKong:
            assert info_tile is not None, '暗杠需要指定牌'
            assert game.kong(current_player_id, info_tile, is_hidden=True), '暗杠失败'
        elif action_type == mj.MessageType.kExtraKong:
            assert game.add_kong(current_player_id, info_tile), '补杠失败'
        else:
            raise Exception('摸牌后只能: 打牌/补杠/暗杠/和')

        return action

    def _execute_after_play(self, game, last_player_id, action):
        player_id = action.player_id
        action_type = action.type
        output_tile = action.output_tile
        info_tile = action.info_tile

        if action_type == mj.MessageType.kWin:
            return player_id
        elif action_type == mj.MessageType.kChow:
            assert (player_id - last_player_id + 4) % 4 == 1, '只能上家吃牌'
            assert game.chow(player_id, info_tile, output_tile), '吃牌失败'
            return (player_id + 1) % 4
        elif action_type == mj.MessageType.kPung:
            assert game.pung(player_id, output_tile), '碰牌失败'
            return (player_id + 1) % 4
        elif action_type == mj.MessageType.kKong:
            assert info_tile is None
            assert game.kong(player_id), '明杠失败'
            return player_id  # 杠后摸牌
        else:
            raise Exception(f'别人打牌后只能: 吃/碰/杠/和 ({action_type})')

    def execute(self, agents, tiles_wall=None, prevalent_wind=None):
        self.agents = agents

        verbose = self.verbose
        game, current_player_id, tiles_wall = self.init_game(tiles_wall, prevalent_wind)

        fan_value = 0
        actions = []
        while tiles_wall:
            last_action = None
            if actions:
                last_action = actions[-1]
                if last_action is None:
                    actions.pop()

            if last_action is not None and verbose:
                print(last_action)
                print(game)

            if last_action is not None and last_action.type == mj.MessageType.kWin:  # 检查是否真的和了
                win_player_id = last_action.player_id
                if verbose:
                    last_message = game.history(game.turn_ID - 1)
                    if last_message.type == mj.MessageType.kDraw:
                        win_tile = last_message.info_tile
                    else:
                        win_tile = last_message.output_tile
                    fan_names = game.win(win_player_id, win_tile)
                    fan_value = print(fan_names)
                else:
                    fan_value = game.try_win(win_player_id)
                    assert fan_value >= 8, '为达到 8 番'

                break

            output_tile = None if last_action is None else last_action.output_tile
            if output_tile is None:
                tile = tiles_wall.pop()
                if is_flower(tile):
                    game.add_flower(current_player_id)
                    agents[current_player_id].add_flower()
                    actions.append(Action(current_player_id, mj.MessageType.kExtraFlower))
                    continue

                # 摸牌并打牌
                actions.append(self._execute_draw(game, current_player_id, tile))
                continue

            # 有玩家打出了牌, 按顺序获得其他玩家的反映
            action = None
            i = (last_action.player_id + 1) % 4
            while i != last_action.player_id:
                agent = self.agents[i]
                response = agent.step(i, game, output_tile=output_tile)  # type: Action
                i = (i + 1) % 4

                if response.type == mj.MessageType.kPass:
                    continue

                if action is None:
                    action = response
                else:
                    old_priority = PRIOIRITY.index(response.type)
                    priority = PRIOIRITY.index(action.type)
                    if priority > old_priority:
                        response = action

            if action is None:  # 下一个人摸牌
                actions.append(None)
                current_player_id = (current_player_id + 1) % 4
            else:
                actions.append(action)
                current_player_id = self._execute_after_play(game, last_action.player_id, action)

        rewards = self.compute_final_rewards(actions[-1].player_id,
                                             actions[-2].player_id, fan_value)

        return fan_value, rewards, actions

    def compute_final_rewards(self, win_player_id, offer_player_id, fan_value):
        rewards = []
        for i in range(4):
            reward = 0
            if fan_value > 0:
                if offer_player_id == win_player_id:  # 自摸
                    if i == win_player_id:
                        reward = 3 * (fan_value + 8)
                    else:
                        reward = -8 - fan_value
                else:
                    if i == win_player_id:
                        reward = fan_value + 24
                    else:
                        reward = -8 - (fan_value if i == offer_player_id else 0)

            rewards.append(reward)
        return rewards
