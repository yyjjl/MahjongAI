# -*- coding: utf-8 -*-

import json
import os

import torch
from dataclasses import dataclass
from tensorboardX import SummaryWriter

from .logging import log_info, log_warning, open_file


@dataclass
class ManagerOptions:
    base_path: str
    checkpoint_freq: int
    log_freq: int


def summary_model(model):
    num_params = 0
    for name, params in model.named_parameters():
        if params.requires_grad:
            num_params += params.numel()
    log_info('params count: %.2fM', num_params / 1024 / 1024)


def optimizer_to_device(op, device):
    for state in op.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def list_checkpoints(checkpoint_dir, include_best=False):
    names = []
    if not os.path.exists(checkpoint_dir):
        return names
    for filename in os.listdir(checkpoint_dir):
        parts = filename.split('.')
        if len(parts) != 3 or parts[0] != 'ckpt' or parts[2] != 'pt':
            continue
        try:
            step = int(parts[1])
        except Exception:
            continue
        names.append((step, os.path.join(checkpoint_dir, filename)))
    checkpoints = [filename for _, filename in sorted(names, reverse=True)]

    extra_paths = [(-1, 'pretrained.pt')]

    if include_best:
        extra_paths.append((0, 'best.pt'))

    for index, path in extra_paths:
        path = os.path.join(checkpoint_dir, path)
        if os.path.exists(path):
            if index == -1:
                index = len(checkpoints)
            checkpoints.insert(index, path)

    return checkpoints


class SessionManager:
    def __init__(self, options: ManagerOptions, training=True):
        self.use_cuda = torch.cuda.is_available()

        self.writer = None
        self.options = options

        self.base_path = options.base_path
        self.checkpoint_freq = options.checkpoint_freq
        self.log_freq = options.log_freq
        self.training = training

        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        self.checkpoint_dir = self.get_path('checkpoints')
        self.summaries_dir = self.get_path('summaries')

        self.metrics_name = None
        self.ckpt_count = 5

    def _restore(self):
        checkpoints = list_checkpoints(self.checkpoint_dir, True)
        assert checkpoints, 'can not find checkpoint files'

        log_info('Session restores from %s', checkpoints[0])

        saved_state = torch.load(checkpoints[0], map_location=self.device)

        self.global_step = saved_state['global_step']
        self.writer = SummaryWriter(log_dir=self.summaries_dir, purge_step=self.global_step)
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

        self.load_state_dict(saved_state)

        self.model.to(self.device)
        optimizer_to_device(self.optimizer, self.device)

        metrics_name = self.metrics_name
        if metrics_name is not None:
            metrics_value = saved_state.get(metrics_name, 0)
            setattr(self, metrics_name, metrics_value)
            log_info('Start from step: %d (%s %.3f)',
                     self.global_step, metrics_name, metrics_value)
        else:
            log_info('Start from step: %d', self.global_step)

    def _init(self):
        assert not list_checkpoints(self.checkpoint_dir), 'checkpoint file(s) already exists !!!'

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.metrics_name is not None:
            setattr(self, self.metrics_name, 0)

        self.global_step = 0
        self.writer = SummaryWriter(log_dir=self.summaries_dir)
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

        self.model.to(self.device)
        optimizer_to_device(self.optimizer, self.device)

        with open_file(os.path.join(self.base_path, 'config'), 'w') as fp:
            json.dump(self.options.__dict__, fp, indent=2)

    @classmethod
    def restore_or_init(cls, path, options_fn, restore=False, **kwargs):
        config_path = os.path.join(path, 'config') if restore else path
        options = options_fn(config_path)
        session_manager = cls(options, **kwargs)

        if restore:
            session_manager._restore()
        else:
            session_manager._init()

        summary_model(session_manager.model)
        return session_manager

    def build_model(self):
        raise NotImplementedError

    def build_optimizer(self):
        raise NotImplementedError

    def state_dict(self, saved_state):
        raise NotImplementedError

    def load_state_dict(self, saved_state):
        raise NotImplementedError

    def close(self):
        self.writer.close()

    def on_update(self, stats, prefix='main'):
        if not self.training:
            return

        self.global_step += 1
        step = self.global_step
        if step % self.log_freq == 0:
            self.log_stats(f'{step}', stats, prefix=prefix)
            self.try_save()

    def log_stats(self, prompt, stats, prefix):
        string = []
        for name, value in stats.items():
            string.append(f'{name}={value:.3f}')
        string = ' ; '.join(string)

        log_info('%s: %s', prompt, string)
        if self.training:
            self.add_summaries(stats.keys(), stats.values(), prefix=prefix)

    def try_save(self, metrics_value=None):
        step = self.global_step
        if step % self.checkpoint_freq != 0:
            if metrics_value is None:
                return
            model_paths = []
        else:
            model_paths = [os.path.join(self.checkpoint_dir, f'ckpt.{self.global_step}.pt')]

        state = {'global_step': step}
        metrics_name = self.metrics_name
        if metrics_name is not None and metrics_value is not None:
            if getattr(self, metrics_name) < metrics_value:
                setattr(self, metrics_name, metrics_value)
                model_paths.append(os.path.join(self.checkpoint_dir, 'best.pt'))
            state[metrics_name] = getattr(self, metrics_name)

        for path in model_paths:
            with open_file(path, 'wb') as fp:
                torch.save(self.state_dict(state), fp)

        for old_checkpoint in list_checkpoints(self.checkpoint_dir)[self.ckpt_count:]:
            try:
                os.remove(old_checkpoint)
            except Exception as err:
                log_warning('%s: can not remove old checkpoint: %s', err, old_checkpoint)

    def add_summaries(self, tags, values, prefix='main', global_step=None):
        if global_step is None:
            global_step = self.global_step
        tags = [tag.replace(' ', '_') for tag in tags]
        self.writer.add_scalars(prefix, dict(zip(tags, values)), global_step)

    def add_summary(self, tag, value, prefix='', global_step=None):
        if not self.training:
            return
        tag = tag.replace(' ', '_')
        if global_step is None:
            global_step = self.global_step
        self.writer.add_scalar(f'{prefix}/{tag}', value, global_step)

    def get_path(self, *name):
        return os.path.join(self.base_path, *name)
