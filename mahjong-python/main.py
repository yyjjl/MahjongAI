# -*- coding: utf-8 -*-

import os

import importlib
import argparse
from framework.logging import log_info


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path_or_restore_dir', type=str)
    parser.add_argument('--model-name', '-m', type=str, default=None)

    cmd_args, _ = parser.parse_known_args(argv)

    model_name = cmd_args.model_name
    module = importlib.import_module(f'models.{model_name}')
    log_info('MODEL_PATH = %s', module.__file__)

    options_fn = module.MainOptions.from_config
    path = cmd_args.config_path_or_restore_dir

    sess_mgr = module.SessionManager.restore_or_init(path, options_fn, restore=os.path.isdir(path))
    dataset = module.DataSet(sess_mgr.options, ['data/mj/MO.*', 'data/mj/PLAY.*'])

    sess_mgr.train(dataset)


if __name__ == '__main__':
    main()
