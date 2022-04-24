from datetime import date

import sys
import logging
import os
import os.path

from .config import Config

_KVARGS_NAME = 'name'
_KVARGS_LOGGER = 'logger'

_args = None

def init(args):
    global _args
    _args = args

# args:
# "name": name of logger which will be create (key: _KVARGS_NAME)
def get_logger(**kvargs) -> logging.Logger:
    global _args

    if Config.Logging.Disable:
        return get_disabled_loggger()

    logdir = Config.Logging.Dir
    if _args is not None and hasattr(_args, 'log_dir'):
        logdir = _args.log_dir

    if len(logdir) > 0 and logdir[len(logdir)-1] != ord('/'):
        logdir += '/'

    filename = f"{logdir}{str(date.today()).replace('-', '_')}_log"
    # create file
    with open(filename, 'w'): "dummy"

    logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
        '%d-%b-%y %H:%M:%S',
    )

    ret_name = ''
    if _KVARGS_NAME in kvargs:
        ret_name = _KVARGS_NAME
    ret = logging.Logger(ret_name)

    fileHandler = logging.FileHandler(filename, mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(_map_log_level())
    ret.addHandler(fileHandler)

    if not Config.Logging.DisableConsole:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(_map_log_level())
        ret.addHandler(consoleHandler)

    return ret

def get_disabled_loggger() -> logging.Logger:
    ret = logging.Logger("disabled")
    fileHandler = logging.FileHandler(os.devnull)
    ret.addHandler(fileHandler)
    return ret

# args:
# "logger": argument for passing logger (key: DO_KVARGS_LOGGER)
def get_logger_via_kvargs(**kvargs) -> logging.Logger:
    if Config.Logging.Disable:
        return get_disabled_loggger()

    ret = logging.root
    if _KVARGS_LOGGER in kvargs:
        ret = kvargs[_KVARGS_LOGGER]
    return ret


def _map_log_level() -> int:
    level = Config.Logging.Level.lower()
    if level == 'debug':
        return logging.DEBUG
    if level == 'info':
        return logging.INFO
    if level in ['warn', 'warning']:
        return logging.WARN
    if level == 'error':
        return logging.ERROR
    if level in ['fatal', 'critical']:
        return logging.FATAL
    raise Exception('Unknown log level in config %s' % Config.Logging.Level)
