from .config import Config

import sys
import logging
import os
import os.path

# Logger: logging.Logger = logging.getLogger('main')
# Logger: logging.Logger = logging.root

def init() -> None:
    if os.path.isfile(Config.Logging.File):
        os.remove(Config.Logging.File)

    # Create a log file
    f = open(Config.Logging.File, 'w')
    f.close()

    logging.root.setLevel(_map_log_level())

    logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
        '%d-%b-%y %H:%M:%S')

    fileHandler = logging.FileHandler(Config.Logging.File, mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(_map_log_level())
    logging.root.addHandler(fileHandler)

    if Config.Logging.Console:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(_map_log_level())
        logging.root.addHandler(consoleHandler)

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

