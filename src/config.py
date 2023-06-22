import pathlib
from typing import Any, List, Union, Type, Dict

import yaml
import numpy as np

# Mark for necessary field
class NecessaryField: pass

# Abstract class for config field classes
class ConfigSection(object):
    # virtual method
    def load(self, cfg: dict) -> None:
        pass

    def yaml(self) -> dict:
        pass

# YAML utils

def _yaml_config_section_arr(arr: List[ConfigSection]):
    return [item.yaml() for item in arr]

def _yaml_optional_config_section(
    cfg: Union[ConfigSection, None],
) -> Union[dict, None]:
    return cfg.yaml() if cfg is not None else None,

def _load_optional_config_section(
    cfg: dict,
    name: str,
    type: Type,
) -> Union[ConfigSection, None]:
    if name not in cfg:
        return None

    ret = type()
    ret.load(Config.get_optional_value(cfg, name, ret))
    return ret


# Config fields class implementations

class LoggingConfigField(ConfigSection):
    Level:          str  = 'info'
    Dir:            str  = ''
    DisableConsole: bool = False
    Disable:        bool = False

    def load(self, cfg: dict) -> None:
        LoggingConfigField.Level   = Config.get_optional_value(cfg, 'level', LoggingConfigField.Level)
        LoggingConfigField.Dir    = Config.get_optional_value(cfg, 'dir', LoggingConfigField.Dir)
        LoggingConfigField.DisableConsole = Config.get_optional_value(cfg, 'disable_console', LoggingConfigField.DisableConsole)
        LoggingConfigField.Disable = Config.get_optional_value(cfg, 'disable', LoggingConfigField.Disable)

    def yaml(self) -> dict:
        return {'level': LoggingConfigField.Level,'dir': LoggingConfigField.Dir,'disable_console': LoggingConfigField.DisableConsole,
                'disable': LoggingConfigField.Disable,}

class DumpConfigField(ConfigSection):
    User:    str  = 'auto'
    Dir:     str  = ''
    Title:   str  = ''
    Disable: bool = False

    def load(self, cfg: dict) -> None:
        DumpConfigField.Disable = Config.get_optional_value(cfg, 'disable', DumpConfigField.Disable)
        DumpConfigField.User   = Config.get_optional_value(cfg, 'user', DumpConfigField.User)
        DumpConfigField.Dir    = Config.get_optional_value(cfg, 'dir', DumpConfigField.Dir)
        DumpConfigField.Title  = Config.get_optional_value(cfg, 'title', DumpConfigField.Title)

    def yaml(self) -> dict:
        return {'title': DumpConfigField.Title, 'disable': DumpConfigField.Disable, 'user': DumpConfigField.User,'dirname': DumpConfigField.Dir,}

class EsnConfigField(ConfigSection):
    @property
    def NReservoir(self) -> int: return self._n_reservoir
    @property
    def SpectralRadius(self) -> float: return self._spectral_radius
    @property
    def Sparsity(self) -> float: return self._sparsity
    @property
    def LambdaR(self) -> float: return self._lambda_r
    @property
    def RandomState(self) -> int: return self._random_state

    @property
    def UseBias(self) -> bool: return self._use_bias
    @property
    def UseAdditiveNoiseWhenForecasting(self) -> bool: return self._use_additive_noise_when_forecasting

    @property
    def InActivation(self) -> str: return self._in_activation
    @property
    def OutActivation(self) -> str: return self._out_activation
    @property
    def Regularization(self) -> str: return self._regularization

    def __init__(self) -> None:
        self._n_reservoir:                         int   = None
        self._spectral_radius:                     float = None
        self._sparsity:                            float = None
        self._lambda_r:                            float = None
        self._random_state:                        int   = 0
        self._in_activation:                       str   = 'tanh'
        self._out_activation:                      str   = 'identity'
        self._regularization:                      str   = 'noise'
        self._use_bias:                            bool  = True
        self._use_additive_noise_when_forecasting: bool  = True

    def load(self, cfg: dict) -> None:
        self._n_reservoir     = Config.get_optional_value(cfg, 'n_reservoir', self._n_reservoir)
        self._spectral_radius = Config.get_optional_value(cfg, 'spectral_radius', self._spectral_radius)
        self._sparsity        = Config.get_optional_value(cfg, 'sparsity', self._sparsity)
        self._lambda_r        = Config.get_optional_value(cfg, 'lambda_r', self._lambda_r)
        self._random_state    = Config.get_optional_value(cfg, 'random_state', self._random_state)
        self._use_bias        = Config.get_optional_value(cfg, 'use_bias', self._use_bias)
        self._in_activation   = Config.get_optional_value(cfg, 'in_activation', self._in_activation)
        self._out_activation  = Config.get_optional_value(cfg, 'out_activation', self._out_activation)
        self._regularization = Config.get_optional_value(cfg, 'regularization', self._regularization)
        self._use_additive_noise_when_forecasting = Config.get_optional_value(cfg, 'use_additive_noise_when_forecasting', self._use_additive_noise_when_forecasting)

    def yaml(self) -> dict:
        return {
            # 'n_inputs': EsnConfigField.NInputs,
            'n_reservoir': self._n_reservoir, 'spectral_radius': self._spectral_radius,
            'sparsity': self._sparsity, 'lambda_r': self._lambda_r,
            'out_activation': self._out_activation, 'in_activation': self._in_activation,
            'regularization': self._regularization, 'random_state': self._random_state,
            'use_bias': self._use_bias, 'use_additive_noise_when_forecasting': self._use_additive_noise_when_forecasting,
        }

class KVArgConfigSection(ConfigSection):
    @property
    def Key(self) -> str: return self._key

    @property
    def Val(self) -> Any: return self._val

    def __init__(self) -> None:
        self._key: str = NecessaryField
        self._val: Any = NecessaryField

    def load(self, cfg: dict) -> None:
        self._key = Config.get_necessary_value(cfg, 'key', self._key)
        self._val = Config.get_necessary_value(cfg, 'val', self._val)

    def yaml(self) -> dict:
        return {
            'key': self._key,'val': self._val,
        }

class EvoOperatorBaseConfigField(ConfigSection):
    @property
    def Method(self) -> str: return self._method

    @property
    def Args(self) -> List[KVArgConfigSection]: return self._args

    def __init__(self) -> None:
        self._method: str                  = NecessaryField
        self._args:   List[KVArgConfigSection] = []

    def load(self, cfg: dict) -> None:
        self._method = Config.get_necessary_value(cfg, 'method', self._method)
        self._args   = Config.get_optional_arr_value(cfg, 'args', KVArgConfigSection, self._args)

    def yaml(self) -> dict:
        return {
            'method': self._method,'args': _yaml_config_section_arr(self._args),
        }

class EvoSelectConfigField(EvoOperatorBaseConfigField):
    def __init__(self) -> None:
        super().__init__()

    def load(self, cfg: dict) -> None:
        super().load(cfg)

    def yaml(self) -> dict:
        ret = super().yaml()
        ret.update()
        return ret

class EvoMateConfigField(EvoOperatorBaseConfigField):
    @property
    def Probability(self) -> float: return self._probability

    def __init__(self) -> None:
        super().__init__()
        self._probability: float = NecessaryField

    def load(self, cfg: dict) -> None:
        super().load(cfg)
        self._probability = Config.get_necessary_value(cfg, 'probability', self._probability)

    def yaml(self) -> dict:
        ret = super().yaml()
        ret.update({
            'probability': self._probability,
        })
        return ret

class EvoMutateConfigField(EvoOperatorBaseConfigField):
    @property
    def Probability(self) -> float: return self._probability

    @property
    def Indpb(self) -> float: return self._indpb

    def __init__(self) -> None:
        super().__init__()
        self._probability: float = NecessaryField
        self._indpb:       float = 0.

    def load(self, cfg: dict) -> None:
        super().load(cfg)
        self._probability = Config.get_necessary_value(cfg, 'probability', self._probability)
        self._indpb       = Config.get_optional_value(cfg, 'indpb', self._indpb)

    def yaml(self) -> dict:
        ret = super().yaml()
        ret.update({
            'probability': self._probability,
            'indpb': self._indpb,
        })
        return ret

class LimitLogspaceConfigField(ConfigSection):
    @property
    def N(self) -> int: return self._n
    @property
    def Power(self) -> int: return self._power

    def __init__(self) -> None:
        self._n:     int = 0
        self._power: int = 10

    def load(self, cfg: dict) -> None:
        self._n     = Config.get_optional_value(cfg, 'n', self._n)
        self._power = Config.get_optional_value(cfg, 'power', self._power)

    def yaml(self) -> dict:
        return {
            'n': self._n,'power': self._power,
        }

class EvoLimitGenConfigField(ConfigSection):
    @property
    def Type(self) -> str: return self._type
    @property
    def Min(self) -> Union[int,float,None]: return self._min
    @property
    def Max(self) -> Union[int,float,None]: return self._max
    @property
    def Choice(self) -> List[Any]: return self._choice
    @property
    def Logspace(self) -> Union[LimitLogspaceConfigField, None]: return self._logspace
    @property
    def Mutate(self) -> Union[EvoOperatorBaseConfigField,None]: return self._mutate
    @property
    def Mate(self) -> Union[EvoOperatorBaseConfigField,None]: return self._mate

    def __init__(self) -> None:
        self._min:        Union[int,float,None]                    = None
        self._max:        Union[int,float,None]                    = None
        self._choice:     List[Any]                                = None
        self._type:       bool                                     = NecessaryField
        self._logspace:   Union[LimitLogspaceConfigField,None]     = None
        self._mutate:     Union[EvoOperatorBaseConfigField,None]   = None
        self._mate:       Union[EvoOperatorBaseConfigField,None]   = None

    def load(self, cfg: dict) -> None:
        self._type       = Config.get_optional_value(cfg, 'type', self._type)
        self._min        = Config.get_optional_value(cfg, 'min', self._min)
        self._max        = Config.get_optional_value(cfg, 'max', self._max)
        self._choice     = Config.get_optional_value(cfg, 'choice', self._choice)

        self._logspace = _load_optional_config_section(cfg, 'logspace', LimitLogspaceConfigField)

        self._mutate = _load_optional_config_section(cfg, 'mutate', EvoOperatorBaseConfigField)
        self._mate   = _load_optional_config_section(cfg, 'mate', EvoOperatorBaseConfigField)

    def yaml(self) -> dict:
        return {
            'type': self._type,
            'min': self._min,'max': self._max,'choice': self._choice,
            'logspace': _yaml_optional_config_section(self._logspace),
            'mutate': _yaml_optional_config_section(self._mutate),
            'mate': _yaml_optional_config_section(self._mate),
        }

class EvoMetricConfigField(ConfigSection):
    @property
    def Name(self) -> str: return self._name
    @property
    def Func(self) -> str: return self._func
    @property
    def Package(self) -> str: return self._package

    @property
    def PltArgs(self) -> List[KVArgConfigSection]: return self._plt_args

    def __init__(self) -> None:
        self._name: str = NecessaryField
        self._func: str = NecessaryField
        # native python math import package
        self._package: str = 'math'

        self._plt_args: List[KVArgConfigSection] = []

    def load(self, cfg: dict) -> None:
        self._name    = Config.get_necessary_value(cfg, 'name', self._name)
        self._func    = Config.get_necessary_value(cfg, 'func', self._func)
        self._package = Config.get_optional_value(cfg, 'package', self._package)

        self._plt_args = Config.get_optional_arr_value(cfg, 'plt_args', KVArgConfigSection, self._plt_args)

    def yaml(self) -> dict:
        return {
            'name': self._name, 'func': self._func, 'package': self._package,
            'plt_args': _yaml_config_section_arr(self._plt_args),
        }

class EvoPopulationConfigField(ConfigSection):
    @property
    def IncludingCount(self) -> int: return self._including_count
    @property
    def HallOfFame(self) -> int: return self._hall_of_fame
    @property
    def MandatoryNewNum(self) -> int: return self._mandatory_new_num

    @property
    def Size(self) -> int: return self._size

    @property
    def Select(self) -> EvoSelectConfigField: return self._select
    @property
    def Mate(self) -> EvoMateConfigField: return self._mate
    @property
    def Mutate(self) -> EvoMutateConfigField: return self._mutate

    @property
    def Limits(self) -> List[EvoLimitGenConfigField]: return self._limits

    def __init__(self) -> None:
        self._including_count:     int = 1
        self._hall_of_fame:        int = 0
        self._mandatory_new_num:   int = 0

        self._size:     int = NecessaryField

        self._select: EvoSelectConfigField = EvoSelectConfigField()
        self._mate:   EvoMateConfigField   = EvoMateConfigField()
        self._mutate: EvoMutateConfigField = EvoMutateConfigField()

        self._limits: List[EvoLimitGenConfigField] = []

    def load(self, cfg: dict) -> None:
        self._size              = Config.get_necessary_value(cfg, 'size', self._size)
        self._including_count   = Config.get_optional_value(cfg, 'including_count', self._including_count)
        self._hall_of_fame      = Config.get_optional_value(cfg, 'hall_of_fame', self._including_count)
        self._mandatory_new_num = Config.get_optional_value(cfg, 'mandatory_new_num', self._mandatory_new_num)

        self._select.load(Config.get_necessary_value(cfg, 'select', self._select))
        self._mate.load(Config.get_necessary_value(cfg, 'mate', self._mate))
        self._mutate.load(Config.get_necessary_value(cfg, 'mutate', self._mutate))

        self._limits = Config.get_optional_arr_value(cfg, 'limits', EvoLimitGenConfigField, self._limits)

    def yaml(self) -> dict:
        return {
            'mandatory_new_num': self._mandatory_new_num,
            'including_count': self._including_count,'hall_of_fame': self._hall_of_fame,'size': self._size,
            'select': self._select.yaml(),'mate': self._mate.yaml(),'mutate': self._mutate.yaml(),
            'limits': _yaml_config_section_arr(self._limits),
        }

class EvoSchemeMultiPopConfigField(ConfigSection):
    @property
    def DumpDir(self) -> str: return self._dump_dir

    @property
    def MaxGenNum(self) -> int: return self._max_gen_num
    @property
    def RandSeed(self) -> int: return self._rand_seed
    @property
    def HromoLen(self) -> int: return self._hromo_len
    @property
    def Verbose(self) -> bool: return self._verbose

    @property
    def FitnessWeights(self) -> List[float]: return self._fitness_weights

    @property
    def Populations(self) -> List[EvoPopulationConfigField]: return self._populations
    @property
    def Metrics(self) -> List[EvoMetricConfigField]: return self._metrics

    def __init__(self) -> None:
        self._dump_dir: str = str(pathlib.Path().resolve())

        self._max_gen_num:     int = NecessaryField
        self._rand_seed:       int = NecessaryField
        self._hromo_len:       int = NecessaryField
        self._verbose:         bool = False

        self._fitness_weights: list[float] = NecessaryField

        self._populations: List[EvoPopulationConfigField] = NecessaryField
        self._metrics:     List[EvoMetricConfigField] = []

    def load(self, cfg: dict) -> None:
        self._dump_dir        = Config.get_optional_value(cfg, 'dump_dir', self._dump_dir)

        self._max_gen_num     = Config.get_necessary_value(cfg, 'max_gen_num', self._max_gen_num)
        self._rand_seed       = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)
        self._hromo_len       = Config.get_necessary_value(cfg, 'hromo_len', self._hromo_len)
        self._verbose         = Config.get_optional_value(cfg, 'verbose', self._verbose)

        self._fitness_weights = Config.get_necessary_value(cfg, 'fitness_weights', self._fitness_weights)

        self._populations = Config.get_necessary_arr_value(cfg, 'populations', EvoPopulationConfigField, self._populations)
        self._metrics     = Config.get_optional_arr_value(cfg, 'metrics', EvoMetricConfigField, self._metrics)

    def yaml(self) -> dict:
        return {
            'dump_dir': self._dump_dir,
            'max_gen_num': self._max_gen_num, 'rand_seed': self._rand_seed, 'hromo_len': self._hromo_len,
            'fitness_weights': self._fitness_weights,'verbose': self._verbose,
            'populations': _yaml_config_section_arr(self._populations),'metrics': _yaml_config_section_arr(self._metrics),
        }

class EvoSchemeConfigField(ConfigSection):
    @property
    def DumpDir(self) -> str: return self._dump_dir

    @property
    def MaxGenNum(self) -> int: return self._max_gen_num
    @property
    def PopulationSize(self) -> int: return self._population_size
    @property
    def RandSeed(self) -> int: return self._rand_seed
    @property
    def HromoLen(self) -> int: return self._hromo_len
    @property
    def HallOfFame(self) -> int: return self._hall_of_fame
    @property
    def MandatoryNewNum(self) -> int: return self._mandatory_new_num

    @property
    def Verbose(self) -> bool: return self._verbose

    @property
    def FitnessWeights(self) -> List[float]: return self._fitness_weights

    @property
    def Select(self) -> EvoSelectConfigField: return self._select
    @property
    def Mate(self) -> EvoMateConfigField: return self._mate
    @property
    def Mutate(self) -> EvoMutateConfigField: return self._mutate

    @property
    def Limits(self) -> List[EvoLimitGenConfigField]: return self._limits
    @property
    def Metrics(self) -> List[EvoMetricConfigField]: return self._metrics

    def __init__(self) -> None:
        self._dump_dir: str = str(pathlib.Path().resolve())

        self._max_gen_num:     int = NecessaryField
        self._population_size: int = NecessaryField
        self._rand_seed:       int = NecessaryField
        self._hromo_len:       int = NecessaryField
        self._hall_of_fame:    int = 0
        self._mandatory_new_num:   int = 0

        self._verbose:         bool = False

        self._fitness_weights: list = NecessaryField

        self._select: EvoSelectConfigField = EvoSelectConfigField()
        self._mate:   EvoMateConfigField   = EvoMateConfigField()
        self._mutate: EvoMutateConfigField = EvoMutateConfigField()

        self._limits: List[EvoLimitGenConfigField] = []
        self._metrics: List[EvoMetricConfigField] = []

    def load(self, cfg: dict) -> None:
        self._dump_dir         = Config.get_optional_value(cfg, 'dump_dir', self._dump_dir)

        self._max_gen_num       = Config.get_necessary_value(cfg, 'max_gen_num', self._max_gen_num)
        self._population_size   = Config.get_necessary_value(cfg, 'population_size', self._population_size)
        self._rand_seed         = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)
        self._hromo_len         = Config.get_necessary_value(cfg, 'hromo_len', self._hromo_len)
        self._hall_of_fame      = Config.get_optional_value(cfg, 'hall_of_fame', self._hall_of_fame)
        self._mandatory_new_num = Config.get_optional_value(cfg, 'mandatory_new_num', self._mandatory_new_num)

        self._verbose         = Config.get_optional_value(cfg, 'verbose', self._verbose)

        self._fitness_weights = Config.get_necessary_value(cfg, 'fitness_weights', self._fitness_weights)

        self._select.load(Config.get_necessary_value(cfg, 'select', self._select))
        self._mate.load(Config.get_necessary_value(cfg, 'mate', self._mate))
        self._mutate.load(Config.get_necessary_value(cfg, 'mutate', self._mutate))

        self._limits = Config.get_optional_arr_value(cfg, 'limits', EvoLimitGenConfigField, self._limits)
        self._metrics = Config.get_optional_arr_value(cfg, 'metrics', EvoMetricConfigField, self._metrics)

    def yaml(self) -> dict:
        return {
            'dump_dir': self._dump_dir, 'mandatory_new_num': self._mandatory_new_num,
            'max_gen_num': self._max_gen_num, 'population_size': self._population_size, 'rand_seed': self._rand_seed, 'hromo_len': self._hromo_len,
            'hall_of_fame': self._hall_of_fame,'fitness_weights': self._fitness_weights,'verbose': self._verbose,
            'select': self._select.yaml(),'mate': self._mate.yaml(),'mutate': self._mutate.yaml(),
            'limits': _yaml_config_section_arr(self._limits),'metrics': _yaml_config_section_arr(self._metrics),
        }

class RandomSearchSchemeSchemeConfigField(ConfigSection):
    @property
    def DumpDir(self) -> str: return self._dump_dir

    @property
    def MaxGenNum(self) -> int: return self._max_gen_num
    @property
    def PopulationSize(self) -> int: return self._population_size
    @property
    def RandSeed(self) -> int: return self._rand_seed
    @property
    def HromoLen(self) -> int: return self._hromo_len
    @property
    def HallOfFame(self) -> int: return self._hall_of_fame

    @property
    def Verbose(self) -> bool: return self._verbose

    @property
    def FitnessWeights(self) -> List[float]: return self._fitness_weights

    @property
    def Limits(self) -> List[EvoLimitGenConfigField]: return self._limits
    @property
    def Metrics(self) -> List[EvoMetricConfigField]: return self._metrics

    def __init__(self) -> None:
        self._dump_dir: str = str(pathlib.Path().resolve())

        self._max_gen_num:       int = NecessaryField
        self._population_size:   int = NecessaryField
        self._rand_seed:         int = NecessaryField
        self._hromo_len:         int = NecessaryField
        self._hall_of_fame:      int = 0

        self._verbose:         bool = False

        self._fitness_weights: list = NecessaryField

        self._limits: List[EvoLimitGenConfigField] = []
        self._metrics: List[EvoMetricConfigField] = []

    def load(self, cfg: dict) -> None:
        self._dump_dir         = Config.get_optional_value(cfg, 'dump_dir', self._dump_dir)

        self._max_gen_num       = Config.get_necessary_value(cfg, 'max_gen_num', self._max_gen_num)
        self._population_size   = Config.get_necessary_value(cfg, 'population_size', self._population_size)
        self._rand_seed         = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)
        self._hromo_len         = Config.get_necessary_value(cfg, 'hromo_len', self._hromo_len)

        self._hall_of_fame    = Config.get_optional_value(cfg, 'hall_of_fame', self._hall_of_fame)
        self._verbose         = Config.get_optional_value(cfg, 'verbose', self._verbose)

        self._fitness_weights = Config.get_necessary_value(cfg, 'fitness_weights', self._fitness_weights)

        self._limits = Config.get_optional_arr_value(cfg, 'limits', EvoLimitGenConfigField, self._limits)
        self._metrics = Config.get_optional_arr_value(cfg, 'metrics', EvoMetricConfigField, self._metrics)

    def yaml(self) -> dict:
        return {
            'dump_dir': self._dump_dir, 'hall_of_fame': self._hall_of_fame,
            'max_gen_num': self._max_gen_num, 'population_size': self._population_size, 'rand_seed': self._rand_seed, 'hromo_len': self._hromo_len,
            'fitness_weights': self._fitness_weights,'verbose': self._verbose,
            'limits': _yaml_config_section_arr(self._limits),'metrics': _yaml_config_section_arr(self._metrics),
        }

class GridSearchSchemeSchemeConfigField(ConfigSection):
    @property
    def DumpDir(self) -> str: return self._dump_dir

    @property
    def MaxGenNum(self) -> int: return self._max_gen_num
    @property
    def PopulationSize(self) -> int: return self._population_size
    @property
    def RandSeed(self) -> int: return self._rand_seed
    @property
    def HromoLen(self) -> int: return self._hromo_len
    @property
    def HallOfFame(self) -> int: return self._hall_of_fame

    @property
    def Verbose(self) -> bool: return self._verbose

    @property
    def FitnessWeights(self) -> List[float]: return self._fitness_weights

    @property
    def Limits(self) -> List[EvoLimitGenConfigField]: return self._limits
    @property
    def Metrics(self) -> List[EvoMetricConfigField]: return self._metrics

    @property
    def GridSplitNum(self) -> List[int]: return self._grid_split_num

    def __init__(self) -> None:
        self._dump_dir: str = str(pathlib.Path().resolve())

        self._max_gen_num:       int = NecessaryField
        self._population_size:   int = NecessaryField
        self._rand_seed:         int = NecessaryField
        self._hromo_len:         int = NecessaryField
        self._hall_of_fame:      int = 0

        self._verbose:         bool = False

        self._fitness_weights: list = NecessaryField

        self._limits:         List[EvoLimitGenConfigField] = NecessaryField
        self._metrics:        List[EvoMetricConfigField]   = []
        self._grid_split_num: List[int]                    = NecessaryField

    def load(self, cfg: dict) -> None:
        self._dump_dir         = Config.get_optional_value(cfg, 'dump_dir', self._dump_dir)

        self._max_gen_num       = Config.get_necessary_value(cfg, 'max_gen_num', self._max_gen_num)
        self._population_size   = Config.get_necessary_value(cfg, 'population_size', self._population_size)
        self._rand_seed         = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)
        self._hromo_len         = Config.get_necessary_value(cfg, 'hromo_len', self._hromo_len)

        self._hall_of_fame    = Config.get_optional_value(cfg, 'hall_of_fame', self._hall_of_fame)
        self._verbose         = Config.get_optional_value(cfg, 'verbose', self._verbose)

        self._fitness_weights = Config.get_necessary_value(cfg, 'fitness_weights', self._fitness_weights)
        self._grid_split_num  = Config.get_necessary_value(cfg, 'grid_split_num', self._grid_split_num)

        self._limits         = Config.get_necessary_arr_value(cfg, 'limits', EvoLimitGenConfigField, self._limits)
        self._metrics        = Config.get_optional_arr_value(cfg, 'metrics', EvoMetricConfigField, self._metrics)

    def yaml(self) -> dict:
        return {
            'dump_dir': self._dump_dir, 'hall_of_fame': self._hall_of_fame,
            'max_gen_num': self._max_gen_num, 'population_size': self._population_size, 'rand_seed': self._rand_seed, 'hromo_len': self._hromo_len,
            'fitness_weights': self._fitness_weights,'verbose': self._verbose,
            'limits': _yaml_config_section_arr(self._limits),'metrics': _yaml_config_section_arr(self._metrics),'grid_split_num': self._grid_split_num,
        }

class ParamLorenzPropField(ConfigSection):
    @property
    def Start(self): return self._start
    @property
    def Stop(self): return self._stop
    @property
    def Num(self): return self._num

    def __init__(self) -> None:
        self._start: float = 0
        self._stop:  float = 0
        self._num:   int   = 0

    def load(self, cfg: dict) -> None:
        self._start = Config.get_optional_value(cfg, 'start', self._stop)
        self._stop  = Config.get_optional_value(cfg, 'stop', self._stop)
        self._num   = Config.get_optional_value(cfg, 'num', self._num)

    def yaml(self) -> dict:
        return {'start': self._start,'stop': self._stop,'num': self._num,}

class LorenzParamsSetPropField(ConfigSection):
    NReservoir:     ParamLorenzPropField = ParamLorenzPropField()
    SpectralRadius: ParamLorenzPropField = ParamLorenzPropField()
    Sparsity:       ParamLorenzPropField = ParamLorenzPropField()
    Noise:          ParamLorenzPropField = ParamLorenzPropField()
    LambdaR:        ParamLorenzPropField = ParamLorenzPropField()

    def load(self, cfg: dict) -> None:
        LorenzParamsSetPropField.NReservoir.load(Config.get_optional_value(cfg, 'n_reservoir', {}))
        LorenzParamsSetPropField.SpectralRadius.load(Config.get_optional_value(cfg, 'spectral_radius', {}))
        LorenzParamsSetPropField.Sparsity.load(Config.get_optional_value(cfg, 'sparsity', {}))
        LorenzParamsSetPropField.Noise.load(Config.get_optional_value(cfg, 'noise', {}))
        LorenzParamsSetPropField.LambdaR.load(Config.get_optional_value(cfg, 'lambda_r', {}))

    def yaml(self) -> dict:
        return {'n_reservoir': LorenzParamsSetPropField.NReservoir.yaml(),'spectral_radius': LorenzParamsSetPropField.SpectralRadius.yaml(),'sparsity': LorenzParamsSetPropField.Sparsity.yaml(),
            'noise': LorenzParamsSetPropField.Noise.yaml(),'lambda_r': LorenzParamsSetPropField.LambdaR.yaml(),}

class ParamsSetGridPropField(ConfigSection):
    Lorenz: LorenzParamsSetPropField = LorenzParamsSetPropField()

    def load(self, cfg: dict) -> None:
        ParamsSetGridPropField.Lorenz.load(Config.get_necessary_value(cfg, 'lorenz', ParamsSetGridPropField.Lorenz))

    def yaml(self) -> dict:
        return {'lorenz': ParamsSetGridPropField.Lorenz.yaml(),}

class GridConfigField(ConfigSection):
    ParamsSet:   ParamsSetGridPropField = ParamsSetGridPropField()

    def load(self, cfg: dict) -> None:
        GridConfigField.ParamsSet.load(Config.get_necessary_value(cfg, 'params_set', GridConfigField.ParamsSet))

    def yaml(self) -> dict:
        return {'params_set': GridConfigField.ParamsSet.yaml(),}

class MoehlisModelConfig(ConfigSection):
    @property
    def Re(self): return self._re
    @property
    def Lx(self): return self._lx_pi
    @property
    def Lz(self): return self._lz_pi
    @property
    def RandSeed(self): return self._rand_seed
    @property
    def Dt(self): return self._dt

    def __init__(self) -> None:
        self._re:        float = 0
        self._lx_pi:     float = 0
        self._lz_pi:     float = 0
        self._rand_seed: int   = 0
        self._dt:        float = 0

    def load(self, cfg: dict) -> None:
        self._re        = Config.get_necessary_value(cfg, 're', self._re)
        self._lx_pi     = Config.get_necessary_value(cfg, 'lx_pi', 0 if self._lx_pi is None else self._lx_pi/np.pi) * np.pi
        self._lz_pi     = Config.get_necessary_value(cfg, 'lz_pi', 0 if self._lz_pi is None else self._lz_pi/np.pi) * np.pi
        self._rand_seed = Config.get_necessary_value(cfg, 'rand_seed', self.RandSeed)
        self._dt        = Config.get_necessary_value(cfg, 'dt', self._dt)

    def yaml(self) -> dict:
        return {
            're': self._re, 'lx_pi': self._lx_pi/np.pi, 'lz_pi': self._lz_pi/np.pi,
            'rand_seed': self.RandSeed, 'dt': self._dt,
        }

class LorenzModelConfig(ConfigSection):
    @property
    def Ro(self): return self._ro
    # @property
    # def RandSeed(self): return self._rand_seed
    @property
    def Dt(self): return self._dt

    def __init__(self) -> None:
        self._ro:        float = 0
        self.RandSeed: int   = 0
        self._dt:        float = 0

    def load(self, cfg: dict) -> None:
        self._ro = Config.get_necessary_value(cfg, 'ro', self._ro)
        self.RandSeed = Config.get_necessary_value(cfg, 'rand_seed', self.RandSeed)
        self._dt = Config.get_necessary_value(cfg, 'dt', self._dt)

    def yaml(self) -> dict:
        return {'ro': self._ro, 'rand_seed': self.RandSeed, 'dt': self._dt,}

class ChuiMoffattModelConfig(ConfigSection):
    # @property
    # def RandSeed(self): return self._rand_seed
    @property
    def Dt(self): return self._dt

    @property
    def Alpha(self): return self._alpha
    @property
    def Omega(self): return self._omega
    @property
    def Eta(self): return self._eta
    @property
    def Kappa(self): return self._kappa
    @property
    def Xi(self): return self._xi

    def __init__(self) -> None:
        self.RandSeed:   int   = NecessaryField
        self._dt:        float = NecessaryField

        self._alpha: float     = NecessaryField
        self._omega: float     = NecessaryField
        self._eta:   float     = NecessaryField
        self._kappa: float     = NecessaryField
        self._xi:    float     = NecessaryField

    def load(self, cfg: dict) -> None:
        self.RandSeed = Config.get_necessary_value(cfg, 'rand_seed', self.RandSeed)
        self._dt      = Config.get_necessary_value(cfg, 'dt', self._dt)
        self._alpha   = Config.get_necessary_value(cfg, 'alpha', self._alpha)
        self._omega   = Config.get_necessary_value(cfg, 'omega', self._omega)
        self._eta     = Config.get_necessary_value(cfg, 'eta', self._eta)
        self._kappa   = Config.get_necessary_value(cfg, 'kappa', self._kappa)
        self._xi      = Config.get_necessary_value(cfg, 'xi', self._xi)

    def yaml(self) -> dict:
        return {
            'rand_seed': self.RandSeed, 'dt': self._dt,
            'alpha': self._alpha,'omega': self._omega,'eta': self._eta,'kappa': self._kappa,'xi': self._xi,
        }

class EsnEvaluateConfigField(ConfigSection):
    @property
    def Metric(self) -> str: return self._metric
    @property
    def MaxSteps(self) -> int: return self._max_steps
    @property
    def N(self) -> int: return self._n
    @property
    def SplitN(self) -> int: return self._split_n
    @property
    def FitStep(self) -> int: return self._fit_step
    @property
    def Normalize(self) -> bool: return self._normalize
    @property
    def Model(self) -> str: return self._model
    @property
    def Data(self) -> str: return self._data

    def __init__(self) -> None:
        self._max_steps: int  = 0
        self._n:         int  = NecessaryField
        self._split_n:   int  = NecessaryField
        self._fit_step:  int  = 0
        self._normalize: bool = False
        self._metric:    str  = NecessaryField
        self._model:     str  = None
        self._data:      str  = None

    def load(self, cfg: dict) -> None:
        self._max_steps = Config.get_optional_value(cfg, 'max_steps', self._max_steps)
        self._n         = Config.get_necessary_value(cfg, 'n', self._n)
        self._split_n   = Config.get_necessary_value(cfg, 'split_n', self._split_n)
        self._fit_step  = Config.get_optional_value(cfg, 'fit_step', self._fit_step)
        self._normalize = Config.get_optional_value(cfg, 'normalize', self._normalize)
        self._metric    = Config.get_necessary_value(cfg, 'metric', self._metric)

        self._model     = Config.get_optional_value(cfg, 'model', self._model)
        self._data      = Config.get_optional_value(cfg, 'data', self._data)
        if self._data is None and self._model is None:
            raise '"model" or "data" fields should be passed to evaluate config'

    def yaml(self) -> dict:
        return {
            'n': self._n,'split_n': self._split_n,'fit_step': self._fit_step,
            'normalize': self._normalize,'max_steps': self._max_steps,'metric': self._metric,
            'model': self._model,'data': self._data,
        }

class DynoEvoEsnHyperParamMultiPopConfig(ConfigSection):
    @property
    def Esn(self) -> EsnConfigField: return self._esn
    @property
    def Evo(self) -> EvoSchemeMultiPopConfigField: return self._evo
    @property
    def Evaluate(self) -> EsnEvaluateConfigField: return self._evaluate

    def __init__(self) -> None:
        self._esn:      EsnConfigField               = EsnConfigField()
        self._evo:      EvoSchemeMultiPopConfigField = EvoSchemeMultiPopConfigField()
        self._evaluate: EsnEvaluateConfigField       = EsnEvaluateConfigField()

    def load(self, cfg: dict) -> None:
        self._esn.load(Config.get_necessary_value(cfg, 'esn', self._esn))
        self._evo.load(Config.get_necessary_value(cfg, 'evo', self._evo))
        self._evaluate.load(Config.get_necessary_value(cfg, 'evaluate', self._evaluate))

    def yaml(self) -> dict:
        return {
            'esn': self._esn.yaml(),'evo': self._evo.yaml(),'evaluate': self._evaluate.yaml(),
        }

class DynoEvoEsnHyperParamConfig(ConfigSection):
    @property
    def Esn(self) -> EsnConfigField: return self._esn
    @property
    def Evo(self) -> EvoSchemeConfigField: return self._evo
    @property
    def Evaluate(self) -> EsnEvaluateConfigField: return self._evaluate

    def __init__(self) -> None:
        self._esn:      EsnConfigField         = EsnConfigField()
        self._evo:      EvoSchemeConfigField   = EvoSchemeConfigField()
        self._evaluate: EsnEvaluateConfigField = EsnEvaluateConfigField()

    def load(self, cfg: dict) -> None:
        self._esn.load(Config.get_necessary_value(cfg, 'esn', self._esn))
        self._evo.load(Config.get_necessary_value(cfg, 'evo', self._evo))
        self._evaluate.load(Config.get_necessary_value(cfg, 'evaluate', self._evaluate))

    def yaml(self) -> dict:
        return {
            'esn': self._esn.yaml(),'evo': self._evo.yaml(),'evaluate': self._evaluate.yaml(),
        }

class HyperParamRandomSearchConfig(ConfigSection):
    @property
    def Esn(self) -> EsnConfigField: return self._esn
    @property
    def RandomSearch(self) -> RandomSearchSchemeSchemeConfigField: return self._random_search
    @property
    def Evaluate(self) -> EsnEvaluateConfigField: return self._evaluate

    def __init__(self) -> None:
        self._esn:           EsnConfigField                      = EsnConfigField()
        self._random_search: RandomSearchSchemeSchemeConfigField = RandomSearchSchemeSchemeConfigField()
        self._evaluate:      EsnEvaluateConfigField              = EsnEvaluateConfigField()

    def load(self, cfg: dict) -> None:
        self._esn.load(Config.get_necessary_value(cfg, 'esn', self._esn))
        self._random_search.load(Config.get_necessary_value(cfg, 'random_search', self._random_search))
        self._evaluate.load(Config.get_necessary_value(cfg, 'evaluate', self._evaluate))

    def yaml(self) -> dict:
        return {
            'esn': self._esn.yaml(),'random_search': self._random_search.yaml(),'evaluate': self._evaluate.yaml(),
        }

class HyperParamGridSearchConfig(ConfigSection):
    @property
    def Esn(self) -> EsnConfigField: return self._esn
    @property
    def GridSearch(self) -> GridSearchSchemeSchemeConfigField: return self._grid_search
    @property
    def Evaluate(self) -> EsnEvaluateConfigField: return self._evaluate

    def __init__(self) -> None:
        self._esn:           EsnConfigField                      = EsnConfigField()
        self._grid_search: GridSearchSchemeSchemeConfigField   = GridSearchSchemeSchemeConfigField()
        self._evaluate:      EsnEvaluateConfigField              = EsnEvaluateConfigField()

    def load(self, cfg: dict) -> None:
        self._esn.load(Config.get_necessary_value(cfg, 'esn', self._esn))
        self._grid_search.load(Config.get_necessary_value(cfg, 'grid_search', self._grid_search))
        self._evaluate.load(Config.get_necessary_value(cfg, 'evaluate', self._evaluate))

    def yaml(self) -> dict:
        return {
            'esn': self._esn.yaml(),'grid_search': self._grid_search.yaml(),'evaluate': self._evaluate.yaml(),
        }

class SchemesConfigField(ConfigSection):
    @property
    def HyperParam(self) -> Union[DynoEvoEsnHyperParamConfig, None]: return self._hyper_param
    @property
    def HyperParamMultiPop(self) -> Union[DynoEvoEsnHyperParamMultiPopConfig, None]: return self._hyper_param_multi_pop
    @property
    def HyperParamMultiPopMultiCrit(self) -> Union[DynoEvoEsnHyperParamMultiPopConfig, None]: return self._hyper_param_multi_pop_multi_crit
    @property
    def HyperParamWithReservoirMultiPop(self) -> Union[DynoEvoEsnHyperParamMultiPopConfig, None]: return self._hyper_param_with_reservoir_multi_pop

    @property
    def HyperParamRandomSearch(self) -> Union[HyperParamRandomSearchConfig, None]: return self._hyper_param_random_search
    @property
    def HyperParamGridSearch(self) -> Union[HyperParamGridSearchConfig, None]: return self._hyper_param_grid_search

    def __init__(self) -> None:
        self._hyper_param:                          Union[DynoEvoEsnHyperParamConfig, None]         = None
        self._hyper_param_multi_pop:                Union[DynoEvoEsnHyperParamMultiPopConfig, None] = None
        self._hyper_param_multi_pop_multi_crit:     Union[DynoEvoEsnHyperParamMultiPopConfig, None] = None
        self._hyper_param_with_reservoir_multi_pop: Union[DynoEvoEsnHyperParamMultiPopConfig, None] = None
        self._hyper_param_random_search:            Union[HyperParamRandomSearchConfig, None]       = None
        self._hyper_param_grid_search:              Union[HyperParamGridSearchConfig, None]         = None

    def load(self, cfg: dict) -> None:
        self._hyper_param                          = _load_optional_config_section(cfg, 'hyper_param', DynoEvoEsnHyperParamConfig)
        self._hyper_param_multi_pop                = _load_optional_config_section(cfg, 'hyper_param_multi_pop', DynoEvoEsnHyperParamMultiPopConfig)
        self._hyper_param_multi_pop_multi_crit     = _load_optional_config_section(cfg, 'hyper_param_multi_pop_multi_crit', DynoEvoEsnHyperParamMultiPopConfig)
        self._hyper_param_with_reservoir_multi_pop = _load_optional_config_section(cfg, 'hyper_param_with_reservoir_multi_pop', DynoEvoEsnHyperParamMultiPopConfig)
        self._hyper_param_random_search            = _load_optional_config_section(cfg, 'hyper_param_random_search', HyperParamRandomSearchConfig)
        self._hyper_param_grid_search              = _load_optional_config_section(cfg, 'hyper_param_grid_search', HyperParamGridSearchConfig)

    def yaml(self) -> dict:
        return {
            'hyper_param': _yaml_optional_config_section(self._hyper_param),
            'hyper_param_multi_pop': _yaml_optional_config_section(self._hyper_param_multi_pop),
            'hyper_param_multi_pop_multi_crit': _yaml_optional_config_section(self._hyper_param_multi_pop_multi_crit),
            'hyper_param_with_reservoir_multi_pop': _yaml_optional_config_section(self._hyper_param_with_reservoir_multi_pop),
            'hyper_param_random_search': _yaml_optional_config_section(self._hyper_param_random_search),
            'hyper_param_grid_search': _yaml_optional_config_section(self._hyper_param_grid_search),
        }

class ModelsConfig(ConfigSection):
    @property
    def Lorenz(self) -> LorenzModelConfig: return self._lorenz
    @property
    def ChuiMoffat(self) -> ChuiMoffattModelConfig: return self._chui_moffatt
    @property
    def Moehlis(self) -> MoehlisModelConfig: return self._moehlis

    def __init__(self) -> None:
        self._lorenz:       LorenzModelConfig      = None
        self._chui_moffatt: ChuiMoffattModelConfig = None
        self._moehlis:      MoehlisModelConfig     = None

    def load(self, cfg: dict) -> None:
        self._lorenz       = _load_optional_config_section(cfg, 'lorenz', LorenzModelConfig)
        self._chui_moffatt = _load_optional_config_section(cfg, 'chui_moffatt', ChuiMoffattModelConfig)
        self._moehlis      = _load_optional_config_section(cfg, 'moehlis', MoehlisModelConfig)

    def yaml(self) -> dict:
        return {
            'lorenz': _yaml_optional_config_section(self._lorenz),
            'chui_moffatt': _yaml_optional_config_section(self._chui_moffatt),
            'moehlis': _yaml_optional_config_section(self._moehlis),
        }

class GlobalPropsConfigField(ConfigSection):
    @property
    def RandSeed(self) -> int: return self._rand_seed

    def __init__(self) -> None:
        self._rand_seed: int = 0

    def load(self, cfg: dict) -> None:
        self._rand_seed = Config.get_optional_value(cfg, 'rand_seed', self._rand_seed)

    def yaml(self) -> dict:
        return {
            'rand_seed': self._rand_seed,
        }

# Main config class

class Config(object):
    GlobalProps: GlobalPropsConfigField = GlobalPropsConfigField()
    Logging:     LoggingConfigField     = LoggingConfigField()
    Dump:        DumpConfigField        = DumpConfigField()
    Models:      ModelsConfig           = ModelsConfig()
    Schemes:     SchemesConfigField     = SchemesConfigField()
    # Grid:      GridConfigField        = GridConfigField()

    def load(cfg: dict, raise_if_necessary: bool=True) -> None:
        Config.__raise_if_necessary = raise_if_necessary

        # Load main sections
        Config.GlobalProps.load(Config.get_optional_value(cfg, 'global_props', Config.GlobalProps))
        Config.Logging.load(Config.get_optional_value(cfg, 'logging', Config.Logging))
        Config.Dump.load(Config.get_optional_value(cfg, 'dump', Config.Dump))
        Config.Models.load(Config.get_necessary_value(cfg, 'models', Config.Models))
        Config.Schemes.load(Config.get_necessary_value(cfg, 'schemes', Config.Schemes))
        # Config.Grid.load(Config.get_necessary_value(cfg, 'grid', Config.Grid))

        Config.__raise_if_necessary = True
        Config.__patched = True

    # Serialization config to yaml
    def yaml() -> dict:
        return {
            # 'grid': Config.Grid.yaml(),
            'global_props': Config.GlobalProps.yaml(),
            'logging': Config.Logging.yaml(), 'dump': Config.Dump.yaml(),
            'models': Config.Models.yaml(),'schemes': Config.Schemes.yaml(),
        }

    # Internal property for control raising exeption if field is not provided
    __raise_if_necessary: bool = True

    # Internal helper func for patching

    def get_optional_value(cfg: dict, key: str, default: Any=None) -> Any:
        if key not in cfg:
            return default
        return cfg[key]

    def get_necessary_value(cfg: dict, key: str, str, default: Any=None) -> Any:
        if key not in cfg:
            if not Config.__raise_if_necessary:
                return default
            raise Exception(f'Failed config load, key \"{key}\" is necessary')
        return cfg[key]

    def get_optional_arr_value(cfg: dict, key: str, bind_item_type: Type=dict, default: Any=None) -> Any:
        if key not in cfg:
            return default
        arr = cfg[key]
        if len(arr) == 0:
            return []

        ret = [0] * len(arr)
        for i in range(len(arr)):
            if issubclass(bind_item_type, ConfigSection):
                ret[i] = bind_item_type()
                ret[i].load(arr[i])
            else:
                ret[i] = arr[i]
        return ret

    def get_necessary_arr_value(cfg: dict, key: str, bind_item_type: Type=dict, default: Any=None) -> Any:
        if key not in cfg:
            if not Config.__raise_if_necessary:
                return default
            raise Exception(f'Failed config load, key \"{key}\" is necessary')
        return Config.get_optional_arr_value(cfg, key, bind_item_type)

    # Pathcing functions

    def patch_from_file(filename: str, raise_if_necessary: bool=True) -> None:
        with open(filename, "r") as config_stream:
            Config.load(yaml.safe_load(config_stream), raise_if_necessary)

    def patch_from_dict(cfg: dict, raise_if_necessary: bool=True) -> None:
        Config.load(cfg, raise_if_necessary)

# Config Interfaces

class ConfigInterface(object):
    def __init__(self) -> None: pass

class IEvoConfig(ConfigInterface):
    @property
    def MaxGenNum(self) -> int: pass
    @property
    def RandSeed(self) -> int: pass
    @property
    def HromoLen(self) -> int: pass

def init(args):
    if args is not None and getattr(args, 'disable_config', False):
        return
    if args is None or not hasattr(args, 'config_path'):
        raise 'cant provide config path'
    if not hasattr(Config, '__patched'):
        Config.patch_from_file(args.config_path)
