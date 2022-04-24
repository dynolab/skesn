from .evo.abstract import Scheme
from .config import Config

from datetime import date

import logging
import dill
import matplotlib.pyplot as plt
import shutil
import yaml
import os
import os.path

FORMAT_FILE = '{0}/frame_{1}_{2}_{3}'
FORMAT_FILE_WITH_TITLE = '{0}/frame_{1}_{2}_{3}_{4}'

def do(scheme: Scheme=None, **kvargs) -> None:
    if not Config.Dump.Enable:
        return

    framedir = create_framedir()

    logging.info(f'dump config... (dir: %s)', framedir)
    # Dump config
    with open(f'{framedir}/config.yaml', 'w') as config_file:
        yaml.safe_dump(Config.yaml(), config_file)

    if Config.Logging.Enable:
        logging.info(f'dump logs... (dir: %s)', framedir)
        # Dump logs
        if os.path.isfile(Config.Logging.File):
            shutil.copyfile(Config.Logging.File, f'{framedir}/log')

    # Dump scheme
    if scheme is not None:
        logging.info(f'dump evo scheme... (dir: %s)', framedir)
        scheme.save(f'{framedir}')

    # Dump kvargs
    do_var(framedir, **kvargs)

created_framedir: str = None

def create_framedir() -> str:
    global created_framedir

    if not Config.Dump.Enable:
        return

    if created_framedir is not None:
        return created_framedir

    if not os.path.isdir(Config.Dump.Dir):
        os.makedirs(Config.Dump.Dir)

    _, dirs, _ = next(os.walk(Config.Dump.Dir))

    max_num = 0
    for dir in dirs:
        if dir.startswith('frame_'):
            num = int(dir.split('_')[-1])
            if num > max_num:
                max_num = num

    framedir = ''
    if Config.Dump.Title == '':
        framedir = FORMAT_FILE.format(
            Config.Dump.Dir,
            str(date.today()).replace('-', '_'),
            Config.Dump.User,
            max_num + 1,
        )
    else:
        framedir = FORMAT_FILE_WITH_TITLE.format(
            Config.Dump.Dir,
            str(date.today()).replace('-', '_'),
            Config.Dump.Title,
            Config.Dump.User,
            max_num + 1,
        )

    os.makedirs(framedir)

    created_framedir = framedir

    return framedir

def do_np_arr(framedir: str=None, **kvargs) -> None:
    if not Config.Dump.Enable:
        return

    if framedir is None:
        framedir = create_framedir()

    for k, v in kvargs.items():
        with open(f'{framedir}/{k}.yaml', 'w') as dump_file:
            if len(v.shape) == 1:
                yaml.safe_dump({k: [float(x) for x in v]}, dump_file)
            elif len(v.shape) == 2:
                dump_mtx = {}
                for i, row in enumerate(v):
                    dump_mtx[f'row_{i}'] = [float(x) for x in row]
                yaml.safe_dump({k: dump_mtx}, dump_file)

def do_var(framedir: str=None, **kvargs) -> None:
    if not Config.Dump.Enable:
        return

    if framedir is None:
        framedir = create_framedir()

    for k, v in kvargs.items():
        if isinstance(v, dict):
            logging.info('try dump "%s" as yaml... (dir: %s)', k, framedir)
            with open(f'{framedir}/{k}.yaml', 'w') as dump_file:
                yaml.safe_dump(v, dump_file)
            continue

        logging.warn('"%s" was not dumped as yaml (instance not dict), try as binary via dill (dir: %s)', k, framedir)
        with open(f'{framedir}/{k}.dill', 'wb') as dump_file:
            dill.dump(v, dump_file)
