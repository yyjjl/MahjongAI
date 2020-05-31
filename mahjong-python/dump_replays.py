# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import pickle

from framework.logging import log_error, open_file
from framework.utils import ProgressReporter
from IO.replay_reader import read_file_content


def _worker(path):
    try:
        replay = read_file_content(path)
        replay['file'] = os.path.basename(path)
        return replay
    except Exception as err:
        if '补花' not in str(err) and 'fans' not in str(err):
            log_error(err, '???')
    return None


def dump_replays(directory, image_size=10000, num_processes=mp.cpu_count()):
    directory = directory.rstrip(os.path.sep)

    image_index = 0
    replays = []

    def _save():
        nonlocal image_index
        nonlocal replays

        path = directory + f'.{image_index}'
        pickle.dump(replays, open_file(path, 'wb'))

        image_index += 1
        replays = []

    total_count = 0
    success_count = 0

    files = os.listdir(directory)
    files = [os.path.join(directory, filename) for filename in files]
    progress = ProgressReporter(len(files), step=1000,
                                message_fn=lambda _: f'{success_count}/{total_count}')
    pool = mp.Pool(processes=num_processes)
    try:
        for replay in progress(pool.imap_unordered(_worker, files, chunksize=100)):
            total_count += 1
            if replay is not None:
                success_count += 1
                replays.append(replay)
                if len(replays) == image_size:
                    print()
                    _save()
        if replays:
            _save()
    finally:
        pool.terminate()


if __name__ == '__main__':
    dump_replays('data/mj//LIU')
    dump_replays('data/mj/MO')
    dump_replays('data/mj/PLAY')
