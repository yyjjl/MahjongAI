# -*- coding: utf-8 -*-

import logging
import os
import sys
import traceback

LOGGER = None
FILE_LOGGER = None

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;{:d}m"
# FMT = '%(levelname)s:%(filename)s: %(message)s'
FMT = '%(levelname)s: %(message)s'
COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': CYAN,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = ''.join([
                COLOR_SEQ.format(30 + COLORS[levelname]),
                levelname,
                RESET_SEQ
            ])
        if record.filename:
            record.filename = ''.join([
                COLOR_SEQ.format(30 + MAGENTA), record.filename, RESET_SEQ
            ])
        return logging.Formatter.format(self, record)


def get_file_logger(log_path=None, level=logging.INFO, name='main.file'):
    global FILE_LOGGER
    if FILE_LOGGER is None:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter(FMT))

        fh = logging.FileHandler(log_path, 'a')
        fh.setFormatter(logging.Formatter(FMT))

        FILE_LOGGER = logging.getLogger(name)
        FILE_LOGGER.setLevel(level)
        FILE_LOGGER.propagate = False
        FILE_LOGGER.addHandler(fh)
        FILE_LOGGER.addHandler(ch)
    return FILE_LOGGER


def get_logger(level=logging.INFO, name='main.console'):
    global LOGGER
    if LOGGER is None:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter(FMT))

        LOGGER = logging.getLogger(name)
        LOGGER.setLevel(level)
        LOGGER.propagate = False
        LOGGER.addHandler(ch)

    return LOGGER


def log_info(fmt, *args, logger=None):
    if logger is None:
        logger = get_logger()
    logger.info(fmt, *args)


def log_warning(fmt, *args, logger=None):
    if logger is None:
        logger = get_logger()
    logger.warning(fmt, *args)


def log_error(err, fmt, *args, logger=None):
    if logger is None:
        logger = get_logger()
    exc_type, ex, tb = sys.exc_info()
    imported_tb_info = traceback.extract_tb(tb)[-1]
    filename = os.path.relpath(imported_tb_info[0], os.path.curdir)
    logger.error('%s@%s#%s:%s: ' + fmt,
                 type(err).__name__, filename, imported_tb_info[1], err, *args)


def open_file(file, mode, **kwargs):
    if mode is None:
        mode = 'r'
    if 'w' in mode:
        char = '>'
    elif 'a' in mode:
        char = '>>'
    else:
        char = '<'
    get_logger().info('%s %s', char, file)

    return open(file, mode, **kwargs)


def make_open_fn(fn):
    def _open(name, mode=None, **kwargs):
        name = fn(name)
        if mode is not None:
            return open_file(name, mode, **kwargs)
        return name

    return _open
