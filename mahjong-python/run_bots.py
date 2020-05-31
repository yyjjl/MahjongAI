# -*- coding: utf-8 -*-

from itertools import combinations, permutations

import pickle
from RL.bots import Arena, KeepRunningBot
from framework.utils import ProgressReporter

ARENA = Arena('data/mahjong-judge', verbose=True)
BOTS = {
    'cnn': ['baseline_cnn.py'],
    # 'cnn2': ['baseline_cnn2.py'],
    'cnn_legacy': ['baseline_cnn_legacy.py'],
    # 'cnn_legacy.512': ['baseline_cnn_legacy.py', 'baseline_cnn_legacy.512.pt'],
    # 'value.weighted': ['value_based.py', 'value_based.weighted.pt'],
    # 'value.probs': ['value_based.py', 'value_based.probs.pt'],
    # 'value.discount': ['value_based.py', 'value_based.discount.pt'],
    'rl_cnn': ['rl_baseline_cnn.py'],
    'random': ['random_bot.py']
}


def init_bots():
    global BOTS

    bots = []
    for name, args in BOTS.items():
        cmd = ['python3']
        cmd.append('../botzone/' + args[0])
        if len(args) > 1:
            cmd.append('data/' + args[1])

        bots.append((cmd, name))

    BOTS = bots


def iter_positions(num_bots):
    for indices in combinations(range(num_bots), 4):
        for positions in permutations(range(4)):
            yield [indices[i] for i in positions]


def main():
    init_bots()

    num_bots = len(BOTS)
    max_games = num_bots * 1000
    total_scores = [0] * num_bots
    total_games = [0] * num_bots

    progress = ProgressReporter(max_games, step=1, print_time=True)
    game_results = []

    args = []
    while len(args) < max_games:
        args.extend(iter_positions(num_bots))

    for num_games, bot_indices in progress(enumerate(args, 1)):
        result = ARENA.execute_one([KeepRunningBot(*BOTS[index]) for index in bot_indices])
        game_results.append(result)
        for index, score in zip(bot_indices, result[0]):
            total_scores[index] += score
            total_games[index] += 1

        if num_games % 20 == 0:
            print(f'[{num_games}/{max_games}] summary:')
            for index in sorted(range(num_bots), key=lambda x: -total_scores[x]):
                print(total_scores[index], total_games[index], BOTS[index][1], sep='\t')

    pickle.dump(game_results, open('data/game_results.pkl', 'wb'))


if __name__ == '__main__':
    main()
