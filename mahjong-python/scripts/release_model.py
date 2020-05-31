# -*- coding: utf-8 -*-

import os
import argparse

import torch

from framework.utils import Options
from framework.session_manager import list_checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--output-path', '-o', type=str)
    parser.add_argument('--list', '-l', action='store_true', default=False)
    parser.add_argument('--lastest', '-L', action='store_true', default=False)
    parser.add_argument('--model-name', '-m', type=str)

    cmd_args, _ = parser.parse_known_args(argv)

    model_dir = cmd_args.model_dir
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')

    if cmd_args.list:
        checkpoints = list_checkpoints(checkpoint_dir, include_best=True)
        print(f'{len(checkpoints)} Find checkpoints in {checkpoint_dir}')
        for checkpoint in checkpoints:
            checkpoint = os.path.basename(checkpoint)
            if checkpoint.endswith('.pt'):
                checkpoint = checkpoint[:-3]
            print('>>>', checkpoint)
        return

    assert cmd_args.output_path is not None, 'output path is empty'

    model_name = cmd_args.model_name
    if model_name is not None:
        checkpoint = None
        for name in [model_name, model_name + '.pt']:
            checkpoint = os.path.join(checkpoint_dir, model_name)
            if os.path.exists(checkpoint):
                break
        assert checkpoint, f'Can not find file {model_name}'
    else:
        checkpoints = list_checkpoints(checkpoint_dir, include_best=not cmd_args.lastest)
        assert checkpoints, f'No checkpoints found in {checkpoint_dir}'

        checkpoint = checkpoints[0]

    print(f'{checkpoint} => {cmd_args.output_path}')
    saved_state = torch.load(checkpoint, map_location='cpu')
    options = Options.from_config(os.path.join(model_dir, 'config'), return_dict=True)
    torch.save({'model': saved_state['model'], 'options': options}, cmd_args.output_path)


if __name__ == '__main__':
    main()
