import numpy as np
import matplotlib.pyplot as plt

import logging

from types import FunctionType
from typing import Any, Dict, List, Tuple

# Constants

# Test config params

TEST_CFG_KEY_NAME              = 'name'
TEST_CFG_KEY_CFG               = 'cfg'
TEST_CFG_KEY_WRAP_EVALUATR_F   = 'wrap_evaluate_f'
TEST_CFG_KEY_WRAP_IND_CRATOR_F = 'wrap_ind_creator_f'
TEST_CFG_KEY_WRAP_EVO_CALLBACK = 'wrap_evo_callback'
TEST_CFG_KEY_VALIDATE_RESULT_F = 'valitate_result_f'
TEST_CFG_KEY_TOOLBOX_SETUP_F   = 'toolbox_setup_f'
TEST_CFG_KEY_DISABLE           = 'disable'

# Default values

DEF_TEST_DUMP_DIR = 'dumps/tests'
DEF_FIG_HEIGHT_INCHES = 8
DEF_FIG_WEIGHT_INCHES = 8

# Utils

def validate_test(test: dict, num: int, logger: logging.Logger) -> bool:
    rules = [
        # {
        #     'msg': f'skip test #{num}, validation error: len args is not in range [2,8]',
        #     'cond': lambda test: len(test) < 2 or len(test) > 8,
        # },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{TEST_CFG_KEY_CFG}"',
            'cond': lambda test: TEST_CFG_KEY_CFG not in test,
        },
        # {
        #     'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_WRAP_IND_CRATOR_F_ARG}"',
        #     'cond': lambda test: _WRAP_IND_CRATOR_F_ARG not in test,
        # },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{TEST_CFG_KEY_WRAP_EVALUATR_F}"',
            'cond': lambda test: TEST_CFG_KEY_WRAP_EVALUATR_F not in test,
        },
        # {
        #     'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_VALIDATE_RESULT_F}"',
        #     'cond': lambda test: _VALIDATE_RESULT_F not in test,
        # },
    ]

    for rule in rules:
        if rule['cond'](test):
            logger.error(rule['msg'])
            return False
    return True

# Radius API functions: iterate 2D points and search points in radius by center points and call callbacks functions on it

def radius_iter_points(
    points: List[Tuple[float,float]],
    centers: List[Tuple[float, float]],
    radius: float,
    *callbacks: List[FunctionType],
) -> None:
    for x, y in points:
        for c_x, c_y in centers:
            r = np.sqrt((x-c_x)**2+(y-c_y)**2)
            if r <= radius:
                for callback in callbacks:
                    callback(x=x, y=y, r=r)

def radius_log_callback(
    logger: logging.Logger,
    pattern: str,
    **kvargs: Dict[str, Any],
) -> FunctionType:
    return lambda x, y, r: logger.info(pattern.format(x=x, y=y, r=r, **kvargs))

def radius_highlight_callback(
    ax: plt.Axes,
    **kvargs: Dict[str, Any],
) -> FunctionType:
    if 'marker' not in kvargs:
        kvargs['marker'] = 'X'
    if 'color' not in kvargs:
        kvargs['color'] = 'green'
    if 'zorder' not in kvargs:
        kvargs['zorder'] = 1
    return lambda x, y, r: ax.scatter(x, y, **kvargs)


_colors = [
    'green', 'red', 'blue', 'black', 'yellow', 'pink', 'gray',
]
_colors_idx = 0

def get_next_color(
    exclude: List=[],
) -> str:
    global _colors_idx

    ret = None

    while True:
        ret = _colors[_colors_idx]
        _colors_idx = (_colors_idx + 1) % len(_colors)
        if ret not in exclude:
            break

    return ret
