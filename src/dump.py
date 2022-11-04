import logging
import shutil
import datetime
from typing import Union
from .config import Config

from datetime import date

import dill
import yaml
import os
import os.path

import src.log as log

_KVARGS_EVO_SCHEME = 'evo_scheme'
_KVARGS_LOGGER = 'logger'

_ALL_KVARGS_KEYS = [
    _KVARGS_EVO_SCHEME,
    _KVARGS_LOGGER,
]

# Arguments for startup program
_dump_rundir: str = None
_args = None

def init(args):
    if Config.Dump.Disable:
        return

    global _args, _dump_rundir
    _args = args

    continue_dir = getattr(_args, 'continue_dir', None)
    if continue_dir is None or len(continue_dir) > 0:
        return
    _dump_rundir = continue_dir

def _get_or_create_dump_rundir(args=None) -> Union[str,None]:
    if Config.Dump.Disable:
        return None

    if args is not None and hasattr(args, 'continue_dir'):
        return getattr(args, 'continue_dir')

    global _dump_rundir
    if _dump_rundir is not None:
        return _dump_rundir

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
        if dir.startswith('run_'):
            num = int(dir.split('_')[-1])
            if num > max_num:
                max_num = num

    dump_rundir = ''
    today = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if Config.Dump.Title == '':
        dump_rundir = '{0}run_{1}_{2}'.format(
            dumpdir,
            max_num + 1,
            today,
        )
    else:
        dump_rundir = '{0}{1}_run_{2}_{3}'.format(
            dumpdir,
            Config.Dump.Title,
            max_num + 1,
            today,
        )

    os.makedirs(dump_rundir)

    _dump_rundir = dump_rundir

    return dump_rundir

# args:
# "evo_scheme": can be passed via kwargs (key: DO_KWARGS_EVO_SCHEME)
# "logger": argument for passing logger (key: DO_KVARGS_LOGGER)
def do(**kvargs) -> None:
    if Config.Dump.Disable:
        return

    # logging.info(f'dump config... (dir: %s)', framedir)
    # # Dump config
    # with open(f'{framedir}/config.yaml', 'w') as config_file:
    #     yaml.safe_dump(Config.yaml(), config_file)

    # if not Config.Logging.Disable:
    #     logging.info(f'dump logs... (dir: %s)', rundir)
    #     logfile = log.get_logfile()
    #     # Dump logs
    #     if os.path.isfile(logfile):
    #         shutil.copyfile(logfile, f'{rundir}/log')

    # Dump scheme
    if _KVARGS_EVO_SCHEME in kvargs:
        rundir = _get_or_create_dump_rundir(_args)

        logging.info(f'dump evo scheme... (dir: %s)', rundir)
        kvargs['evo_scheme'].save(f'{rundir}')

    # Dump kvargs
    do_var(**kvargs)

# args:
def do_np_arr(**kvargs) -> None:
    if Config.Dump.Disable:
        return

    framedir = _get_or_create_dump_rundir(_args)

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

    framedir = _get_or_create_dump_rundir(_args)

    for k, v in kvargs.items():
        if k in _ALL_KVARGS_KEYS:
            continue

        if isinstance(v, dict):
            logging.info('try dump "%s" as yaml... (dir: %s)', k, framedir)
            with open(f'{framedir}/{k}.yaml', 'w') as dump_file:
                yaml.safe_dump(v, dump_file)
            continue

        logging.warn('"%s" was not dumped as yaml (instance not dict), try as binary via dill (dir: %s)', k, framedir)
        with open(f'{framedir}/{k}.dill', 'wb') as dump_file:
            dill.dump(v, dump_file)
