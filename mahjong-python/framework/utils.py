# -*- coding: utf-8 -*-

import json
import os
import sys
import time


def override(x):
    return x


class AverageMeter:
    def __init__(self):
        self._data = {}

    def __getattr__(self, key):
        if key in self._data:
            value, count = self._data[key]
            return value / count
        raise AttributeError

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return object.__setattr__(self, key, value)
        self._data[key] = value, 0

    def items(self):
        return zip(self.keys(), self.values())

    def clear(self):
        for key in self._data:
            self._data[key] = 0, 0

    def keys(self):
        return self._data.keys()

    def values(self):
        return [value / count for value, count in self._data.values()]

    def add(self, key, value, count=1):
        if count == 0:
            return
        old_value, old_count = self._data.get(key, (0, 0))
        self._data[key] = old_value + value, old_count + count


class ProgressReporter:
    def __init__(self, stop=-1, step=None,
                 message_fn=None, prompt='progress',
                 stream=sys.stderr, print_time=False, newline=False):
        self.current = 0
        self.stop = stop
        if step is None:
            step = max(10, stop // 5)
        self._step = step
        self._stream = stream
        self._fn = message_fn
        self._prompt = prompt
        self._print_time = print_time
        self._printed = False
        self._newline = newline

    def _report(self):
        try:
            suffix = self._fn(self) if self._fn else ''
        except Exception:
            suffix = '<error occurs>'
        if self._print_time and self.current != 0:
            speed = ' ({:.3f}s/tick)'.format((time.time() - self._start_time) / self.current)
        else:
            speed = ''
        stop = '?' if self.stop < 1 else self.stop
        message = f'{self._prompt}: {self.current}/{stop}{speed} {suffix}'
        self._stream.write(message)
        self._stream.write('\n' if self._newline else '\r')

    def __enter__(self, *_):
        self.current = 0
        if self._print_time:
            self._start_time = time.time()
        return self

    def __exit__(self, *_):
        if not self._printed:
            self._report()
        if not self._newline:
            self._stream.write('\n')

    def __iter__(self):
        with self:
            for i in range(0, self.stop):
                yield i
                self.tick()

    def __call__(self, iterable):
        with self:
            for value in iterable:
                yield value
                self.tick()

    def tick(self, count=1):
        self.current += count
        if self.current % self._step == 0:
            self._printed = True
            self._report()
        else:
            self._printed = False


class Options:
    @classmethod
    def from_config(cls, config_path, return_dict=False):
        if return_dict:
            cls = dict

        try:
            with open(config_path) as fp:
                options = json.load(fp)
            return cls(**options)
        except json.JSONDecodeError:
            pass
        options = {}
        visited_paths = set()

        def _load(path):
            if path in visited_paths:
                return

            visited_paths.add(path)
            cur_dir = os.path.dirname(path)

            def _include(base_path):
                _load(os.path.join(cur_dir, base_path))
            with open(path) as fp:
                code = fp.read()
            exec(compile(code, path, 'exec'), {'import_config': _include}, options)

        try:
            _load(os.path.realpath(os.path.abspath(config_path)))
        except Exception as err:
            raise Exception(f'Failed to load {config_path}: {err}')

        return cls(**options)


def query_yes_no(question, default='yes', with_choice_all=False):
    """The "answer" return value is True for "yes" or False for "no"."""
    valid = {'yes': 'yes', 'y': 'yes', 'no': 'no', 'n': 'no'}
    if default is None:
        prompt = '[y/n{}] '
    elif default == 'yes':
        prompt = '[Y/n{}] '
    elif default == 'no':
        prompt = '[y/N{}] '
    else:
        raise ValueError('invalid default answer: "{}"'.format(default))

    prompt = prompt.format('/!y/!n' if with_choice_all else '')
    if with_choice_all:
        valid.update({'!yes': 'all-yes', '!y': 'all-yes', '!no': 'all-no', '!n': 'all-no'})
    while True:
        print(question, prompt, end='')
        choice = input().lower().strip()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print('Please respond with \'yes\' or \'no\' ', end='')


def iter_batch_spans(size, batch_size):
    for i in range((size + batch_size - 1) // batch_size):
        yield i * batch_size, min(size, (i + 1) * batch_size)
