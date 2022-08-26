from ast import arg
from .log import get_logger_via_kvargs
from .config import Config

from datetime import date

import dill
import yaml
import os
import os.path

_KVARGS_EVO_SCHEME = 'evo_scheme'
_KVARGS_LOGGER = 'logger'

_ALL_KVARGS_KEYS = [
    _KVARGS_EVO_SCHEME,
    _KVARGS_LOGGER,
]

# Arguments for startup program
_args = None

def init(args):
    global _args
    _args = args

_created_framedir: str = None

def _create_framedir(args=None) -> str:
    global _created_framedir

    if _created_framedir is not None:
        return _created_framedir

    dumpdir = ''
    if args is not None and hasattr(args, 'dump_dir'):
        dumpdir = args.dump_dir
    else:
        dumpdir = Config.Dump.Dir

    if len(dumpdir) > 0 and ord(dumpdir[len(dumpdir)-1]) != ord('/'):
        dumpdir += '/'

    if not os.path.isdir(dumpdir):
        os.makedirs(dumpdir)

    _, dirs, _ = next(os.walk(dumpdir))

    max_num = 0
    for dir in dirs:
        if dir.startswith('frame_'):
            num = int(dir.split('_')[-1])
            if num > max_num:
                max_num = num

    framedir = ''
    if Config.Dump.Title == '':
        framedir = '{0}frame_{1}_{2}_{3}'.format(
            dumpdir,
            str(date.today()).replace('-', '_'),
            Config.Dump.User,
            max_num + 1,
        )
    else:
        framedir = '{0}frame_{1}_{2}_{3}_{4}'.format(
            dumpdir,
            str(date.today()).replace('-', '_'),
            Config.Dump.Title,
            Config.Dump.User,
            max_num + 1,
        )

    os.makedirs(framedir)

    _created_framedir = framedir

    return framedir

# args:
# "evo_scheme": can be passed via kwargs (key: DO_KWARGS_EVO_SCHEME)
# "logger": argument for passing logger (key: DO_KVARGS_LOGGER)
def do(**kvargs) -> None:
    if Config.Dump.Disable:
        return

    logger = get_logger_via_kvargs(**kvargs)

    framedir = _create_framedir(_args)

    logger.info(f'dump config... (dir: %s)', framedir)
    # Dump config
    with open(f'{framedir}/config.yaml', 'w') as config_file:
        yaml.safe_dump(Config.yaml(), config_file)

    # if not Config.Logging.Disable:
    #     logging.info(f'dump logs... (dir: %s)', framedir)
    #     # Dump logs
    #     if os.path.isfile(Config.Logging.Dir):
    #         shutil.copyfile(Config.Logging.Dir, f'{framedir}/log')

    # Dump scheme

    if _KVARGS_EVO_SCHEME in kvargs:
        logger.info(f'dump evo scheme... (dir: %s)', framedir)
        kvargs['evo_scheme'].save(f'{framedir}')

    # Dump kvargs
    do_var(framedir, **kvargs)

# args:
def do_np_arr(**kvargs) -> None:
    if Config.Dump.Disable:
        return

    framedir = _create_framedir(_args)

    for k, v in kvargs.items():
        if k in _ALL_KVARGS_KEYS:
            continue

        with open(f'{framedir}/{k}.yaml', 'w') as dump_file:
            if len(v.shape) == 1:
                yaml.safe_dump({k: [float(x) for x in v]}, dump_file)
            elif len(v.shape) == 2:
                dump_mtx = {}
                for i, row in enumerate(v):
                    dump_mtx[f'row_{i}'] = [float(x) for x in row]
                yaml.safe_dump({k: dump_mtx}, dump_file)

# args:
# "logger": argument for passing logger (key: DO_KVARGS_LOGGER)
def do_var(**kvargs) -> None:
    if Config.Dump.Disable:
        return

    framedir = _create_framedir(_args)

    logger = get_logger_via_kvargs(**kvargs)

    for k, v in kvargs.items():
        if k in _ALL_KVARGS_KEYS:
            continue

        if isinstance(v, dict):
            logger.info('try dump "%s" as yaml... (dir: %s)', k, framedir)
            with open(f'{framedir}/{k}.yaml', 'w') as dump_file:
                yaml.safe_dump(v, dump_file)
            continue

        logger.warn('"%s" was not dumped as yaml (instance not dict), try as binary via dill (dir: %s)', k, framedir)
        with open(f'{framedir}/{k}.dill', 'wb') as dump_file:
            dill.dump(v, dump_file)
