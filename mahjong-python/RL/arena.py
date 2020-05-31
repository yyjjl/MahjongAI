# -*- coding: utf-8 -*-

import PyMahjong as mj
import random

from IO.replay_reader import Action

ALL_TILES = list(range(1, 35)) * 4 + list(range(100, 108))
PRIOIRITY = [mj.MessageType.kWin, mj.MessageType.kKong, mj.MessageType.kPung, mj.MessageType.kChow]


def is_flower(tile):
    return tile >= 100


def copy_game_setup(setup):
    return [(_.copy() if hasattr(_, 'copy') else _) for _ in setup]


def create_game():
    game = mj.Game()

    prevalent_wind = random.randrange(4)
    master_id = random.randrange(4)

    game.init_seat(mj.Wind(prevalent_wind), master_id)

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

    for i in range(4):
        assert game.init_tiles(i, initial_tiles[i], flower_counts[i]), '初始化失败'

    return game, master_id, tiles_wall


def print_fan_names(fan_names):
    print(fan_names)
    return sum(value for _, value in fan_names)


class Arena:
    def __init__(self, verbose=False, use_extra_reward=True):
        self.verbose = verbose
        self.use_extra_reward = use_extra_reward

    def compute_score(self, game, current_player_id):
        if not self.use_extra_reward:
            return 0

        player = game.player(current_player_id)
        rest_counts = game.view_of_player(current_player_id)[0].reshape((-1,))[:34]

        n1, n2, n3, useful_tiles = player.shanten()
        score = -min(n1, n2, n3) + rest_counts[useful_tiles].sum() / 136.0

        return score

    def _execute_draw(self, game, current_player_id, tile):
        game.draw(current_player_id, tile)

        score_before = self.compute_score(game, current_player_id)

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

        score_after = self.compute_score(game, current_player_id)
        agent.add_reward(current_player_id, score_after - score_before)

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

    def execute(self, agents, game_setup=None):
        self.agents = agents

        if game_setup is None:
            game_setup = create_game()

        verbose = self.verbose
        game, current_player_id, tiles_wall = game_setup

        for agent in self.agents:
            agent.reset()

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
                    fan_value = print_fan_names(fan_names)
                else:
                    fan_value = game.try_win(win_player_id)
                    assert fan_value >= 8, '为达到 8 番'
                fan_value -= game.player(win_player_id).flower_count  # 减去花牌

                break

            # TODO: 考虑抢扛和

            output_tile = None if last_action is None else last_action.output_tile
            if output_tile is None:
                tile = tiles_wall.pop()
                if is_flower(tile):  # 补花不需要 agent 取考虑
                    game.add_flower(current_player_id)
                    actions.append(Action(current_player_id, mj.MessageType.kExtraFlower))
                    continue

                # 摸牌并打牌
                actions.append(self._execute_draw(game, current_player_id, tile))
                continue

            # 有玩家打出了牌, 按顺序获得其他玩家的反映
            action = None
            score_befores = []
            i = (last_action.player_id + 1) % 4
            while i != last_action.player_id:
                agent = self.agents[i]

                score_befores.append((i, self.compute_score(game, i)))

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

            for i, score_before in score_befores:
                score_after = self.compute_score(game, i)
                self.agents[i].add_reward(i, score_after - score_before)

        self.compute_final_rewards(actions[-1].player_id, actions[-2].player_id, fan_value)

        return fan_value, actions

    def compute_final_rewards(self, win_player_id, offer_player_id, fan_value):
        for i, agent in enumerate(self.agents):
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

            agent.finish(i, reward)
