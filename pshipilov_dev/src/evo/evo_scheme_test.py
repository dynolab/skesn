import logging
from ..log import get_logger

from pshipilov_dev.src.config import EvoSchemeConfigField
from pshipilov_dev.src.evo.evo_scheme import EvoScheme

import numpy as np
from deap import creator

from typing import List, Tuple

# Test 1 implementations

def _test1_wrap_ind_creator_f(cfg: EvoSchemeConfigField):
    def _test1_ind_creator_f():
        ret = [0] * cfg.HromoLen
        for i in range(cfg.HromoLen):
            ret[i] = np.random.randint(2)
        return creator.Individual(ret)
    return _test1_ind_creator_f

def _test1_wrap_evaluate_f(cfg: EvoSchemeConfigField):
    def _test1_evaluate_f(ind: list) -> Tuple[int]:
        ret = 0
        for x in ind:
            ret += x
        return ret,
    return _test1_evaluate_f

def _test1_validate_result_f(cfg: EvoSchemeConfigField, last_popultaion: list) -> Tuple[bool, List]:
    expected = cfg.HromoLen
    for ind in last_popultaion:
        actual = 0
        for x in ind:
            actual += x
        if actual == expected:
            return True, ind
    return False, None
# {
#             'rand_seed': 1,
#             'max_gen_num': 200,
#             'population_size': 50,
#             'fitness_weights': [1.0,],

#             'select': {
#                 'method': 'selTournament',
#                 'args': [
#                     {'key': 'tournsize','val': 3},
#                 ],
#             },
#             'mate': {
#                 'method': 'cxSimulatedBinaryBounded',
#                 'probability': 0.9,
#                 'args': [
#                     {'key': 'eta','val': 20.0},
#                 ],
#             },
#             'mutate': {
#                 'method': 'mutPolynomialBounded',
#                 'probability': 0.15,
#                # 'indpb': 0.25,
#             },

#             'limits': [
#                 {'min': 0.2,'max': 1.2},
#                 {'min': 0.001,'max': 0.8},
#                 {'min': 0.00000001,'max': 0.01},
#             ],
#         },
#         'ind_creator_f': _test1_ind_creator_f,
#         'evaluate_f': _test1_evaluate_f,
#     }

_NAME_ARG = 'name'
_CFG_ARG = 'cfg'
_WRAP_IND_CRATOR_F_ARG = 'wrap_ind_creator_f'
_WRAP_EVALUATR_F_ARG = 'wrap_evaluate_f'
_VALIDATE_RESULT_F_ARG = 'valitate_result_f'

_TESTS = [
    {
        _NAME_ARG: 'one_max',
        _CFG_ARG: {
            'rand_seed': 1,
            'max_gen_num': 200,
            'population_size': 50,
            'hromo_len': 100,

            'fitness_weights': [1.0,],

            'select': {
                'method': 'selTournament',
                'args': [
                    {'key': 'tournsize','val': 3},
                ],
            },
            'mate': {
                'method': 'cxOnePoint',
                'probability': 0.9,
            },
            'mutate': {
                'method': 'mutFlipBit',
                'probability': 0.1,
               # 'indpb': 0.25,
            },
        },
        _WRAP_IND_CRATOR_F_ARG: _test1_wrap_ind_creator_f,
        _WRAP_EVALUATR_F_ARG: _test1_wrap_evaluate_f,
        _VALIDATE_RESULT_F_ARG: _test1_validate_result_f,
    }
]

def _validate_test(test: dict, num: int, logger: logging.Logger) -> bool:
    rules = [
        {
            'msg': f'skip test #{num}, validation error: len args is not equal 4',
            'cond': lambda test: len(test) != 4 and len(test) != 5,
        },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_CFG_ARG}"',
            'cond': lambda test: _CFG_ARG not in test,
        },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_WRAP_IND_CRATOR_F_ARG}"',
            'cond': lambda test: _WRAP_IND_CRATOR_F_ARG not in test,
        },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_WRAP_EVALUATR_F_ARG}"',
            'cond': lambda test: _WRAP_EVALUATR_F_ARG not in test,
        },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_VALIDATE_RESULT_F_ARG}"',
            'cond': lambda test: _VALIDATE_RESULT_F_ARG not in test,
        },
    ]

    for rule in rules:
        if rule['cond'](test):
            logger.error(rule['msg'])
            return False
    return True

def run_tests():
    logger = get_logger(name='tests')

    for i, test in enumerate(_TESTS):
        if not _validate_test(test, i, logger):
            continue

        # Prepare

        name = test[_NAME_ARG] if _NAME_ARG in test else f'test_#{i}'
        cfg = EvoSchemeConfigField()
        cfg.load(test[_CFG_ARG])
        scheme = EvoScheme(
            name,
            cfg,
            test[_WRAP_IND_CRATOR_F_ARG](cfg),
            test[_WRAP_EVALUATR_F_ARG](cfg),
        )

        # Action
        last_popultaion = scheme.run()

        # Assert
        ok, sol = test[_VALIDATE_RESULT_F_ARG](cfg, last_popultaion)
        if ok:
            logger.info('test #%d - successful (one of solution: [%s])', i, ','.join(map(str, sol)))
        else:
            logger.error('test #%d - wrong', i)

