# -*- coding: utf-8 -*-

from typing import Optional

from dataclasses import dataclass

import PyMahjong as mj

SPEICAL_FAN_NAMES = {'七星不靠', '全不靠', '组合龙'}

TILE_NAMES = [
    '???',
    'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
    'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3'
]
TILE_NAME2INDEX = dict(map(reversed, enumerate(TILE_NAMES)))

WINDS = {
    '东': mj.Wind.kEast,
    '西': mj.Wind.kWest,
    '南': mj.Wind.kSouth,
    '北': mj.Wind.kNorth,
}

SUITS = {
    'T': mj.TILE_SUIT_BAMBOO,
    'B': mj.TILE_SUIT_DOTS,
    'W': mj.TILE_SUIT_CHARACTERS,
    'F': mj.TILE_SUIT_HONORS,
    'J': mj.TILE_SUIT_HONORS
}

NAMES = {
    '坎张': '嵌张',
    '断幺九': '断幺'
}


@dataclass
class Action:
    player_id: int
    type: mj.MessageType
    info_tile: Optional[str] = None
    output_tile: Optional[str] = None
    offer: Optional[int] = None


def tile_to_index(tile):
    return TILE_NAME2INDEX.get(tile, -1)


def tiles_to_indices(tiles):
    return [TILE_NAME2INDEX.get(tile, -1) for tile in tiles]


def _read_raw_actions(fp):
    raw_actions = []
    for line in fp:
        player_id, action, args, *extra_args = line.split()
        player_id = int(player_id)
        args = tiles_to_indices(eval(args))
        extra_args = [int(arg) if arg[0].isnumeric() else tile_to_index(arg)
                      for arg in extra_args]

        raw_actions.append([player_id, action, args, extra_args])
    return raw_actions


def _read_actions(fp):
    actions = []
    raw_actions = _read_raw_actions(fp)
    raw_actions.reverse()

    while raw_actions:
        player_id, action, args, extra_args = raw_actions.pop()

        info_tile = None
        output_tile = None
        offer = None
        if action == '打牌':
            assert not extra_args and len(args) == 1
            output_tile = args[0]
            type = mj.MessageType.kPlay
        elif '摸牌' in action:
            assert not extra_args and len(args) == 1, '??? 摸牌'

            info_tile = args[0]
            if info_tile == -1:  # 补花
                next_player_id, next_action, next_args, next_extra_args = raw_actions.pop()
                assert next_player_id == player_id \
                    and next_action == '补花' \
                    and not next_extra_args \
                    and next_args == args, '??? 补花'

                type = mj.MessageType.kExtraFlower
            else:
                type = mj.MessageType.kDraw
        elif action == '吃':
            assert len(args) == 3 and len(extra_args) == 2 \
                and extra_args[0] == actions[-1].output_tile \
                and extra_args[1] == actions[-1].player_id, '??? 吃'

            next_player_id, next_action, next_args, next_extra_args = raw_actions.pop()

            assert next_player_id == player_id \
                and next_action == '打牌' \
                and len(next_args) == 1 and not next_extra_args, '??? 吃后打牌'

            type = mj.MessageType.kChow
            info_tile = args[1]
            output_tile = next_args[0]
        elif action == '碰':
            assert len(args) == 3 and len(extra_args) == 2 \
                and extra_args[0] == actions[-1].output_tile \
                and extra_args[1] == actions[-1].player_id, '??? 碰'

            next_player_id, next_action, next_args, next_extra_args = raw_actions.pop()

            assert next_player_id == player_id \
                and next_action == '打牌' \
                and len(next_args) == 1 and not next_extra_args, '??? 碰后打牌'

            type = mj.MessageType.kPung
            output_tile = next_args[0]
            offer = mj.Wind(extra_args[1])
        elif action == '和牌':
            if raw_actions:   # 最后和牌, 中间和牌报错
                raise Exception('fans <= 8')
            type = mj.MessageType.kWin
            info_tile = args[0]
        elif action == '明杠':
            assert len(args) == 4 and len(extra_args) == 2 \
                and extra_args[0] == actions[-1].output_tile \
                and extra_args[1] == actions[-1].player_id, '??? 明杠'

            type = mj.MessageType.kKong
            offer = mj.Wind(extra_args[1])
        elif action == '补杠':
            # 这里规则有点不太一样, 任何时刻可以补杠
            assert len(args) == 4 and len(extra_args) == 2

            type = mj.MessageType.kExtraKong
            info_tile = args[0]
        elif action == '暗杠':
            # 这里规则有点不太一样, 任何时刻可以暗杠
            assert len(args) == 4 and len(extra_args) == 2

            type = mj.MessageType.kKong
            info_tile = args[0]
        else:
            raise Exception(f'Unknown action: {action}')

        actions.append(Action(player_id=player_id,
                              type=type,
                              info_tile=info_tile,
                              output_tile=output_tile,
                              offer=offer))

    return actions


def read_file_content(path):
    with open(path) as fp:
        next(fp)  # skip xml name

        wind, score, fans, tag = next(fp).split()
        fan_names = {}
        for fan in eval(fans):
            name, *value = fan.split('-')
            if not value:
                value = [-1]
            value = int(value[0])
            name = NAMES.get(name, name)
            if value == 0:
                continue
            fan_names[name] = fan_names.get(name, 0) + value

        replay = {
            'wind': WINDS[wind],
            'score': int(score),
            'fan_names': fan_names,
            'tag': tag
        }

        initial_tiles = [None] * 4
        for _ in range(4):
            player_id, tiles, flower_count = next(fp).split()
            initial_tiles[int(player_id)] = tiles_to_indices(eval(tiles)), int(flower_count)

        replay['initial_tiles'] = initial_tiles
        replay['actions'] = _read_actions(fp)

    return replay
