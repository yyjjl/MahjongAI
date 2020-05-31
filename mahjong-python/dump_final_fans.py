# -*- coding: utf-8 -*-

import glob
import pickle

import PyMahjong as mj
from framework.logging import log_error, open_file
from models.baseline_cnn.dataloader import Loader as LoaderBase
from models.baseline_cnn.dataloader import load_replays, DataSetOptions


class Loader(LoaderBase):
    def worker(self, replay):
        try:
            actions = replay['actions']
            final_action = actions[-1]
            score = replay['score']
            if final_action.type != mj.MessageType.kWin:
                return None

            player_id = final_action.player_id
            result = self.simulate(replay, True)
            if result is None:
                return None
            game = result[0]
            score -= game.player(player_id).flower_count  # 减去花数

            counts = [x.reshape((-1,)[:34]) for x in game.view_of_player(player_id)]
            fix_packs = game.all_packs(player_id)[0][:4]

            if game.history(game.turn_ID - 1).type != mj.MessageType.kDraw:
                counts[1][final_action.info_tile - 1] += 1
                counts[2][final_action.info_tile - 1] += 1

            return counts, fix_packs, score
        except Exception as err:
            log_error(err, '%s', replay['file'])
            return None


def main():
    loader = Loader(DataSetOptions(True, True, 64))

    examples = []
    for pattern in ['data/mj/PLAY.*', 'data/mj/MO.*']:
        for file in glob.glob(pattern):
            examples.extend(load_replays(loader, file))

    pickle.dump(examples, open_file('data/mj/final_fans.pkl', 'wb'))


if __name__ == '__main__':
    main()
