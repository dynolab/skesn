from typing import Any, List, Union, Type, Dict

import yaml

# Mark for necessary field
class NecessaryField: pass

# Abstract class for config field classes
class ConfigSection(object):
    # virtual method
    def load(self, cfg: dict) -> None:
        pass

    def yaml(self) -> dict:
        pass

# Config fields class implementations

def _yaml_config_section_arr(arr: List[ConfigSection]):
    return [item.yaml() for item in arr]

class EvaluateOptsConfigField(ConfigSection):
    @property
    def SparsityTrain(self) -> int: return self._sparsity_train

    def __init__(self) -> None:
        self._sparsity_train: int = 0

    def load(self, cfg: dict) -> None:
        self._sparsity_train = Config.get_optional_value(cfg, 'sparsity_train', self._sparsity_train)

    def yaml(self) -> dict:
        return {'sparsity_train': self._sparsity_train,}

class EvaluateConfigField(ConfigSection):
    @property
    def Metric(self) -> str: return self._metric
    @property
    def Model(self) -> str: return self._model
    @property
    def Steps(self) -> int: return self._steps
    @property
    def Opts(self) -> EvaluateOptsConfigField: return self._opts

    def __init__(self) -> None:
        self._metric:  str                   = NecessaryField
        self._model: str                     = NecessaryField
        self._steps: int                     = 1
        self._opts:  EvaluateOptsConfigField = EvaluateOptsConfigField()

    def load(self, cfg: dict) -> None:
        self._metric  = Config.get_necessary_value(cfg, 'metric', self._metric)
        self._model = Config.get_necessary_value(cfg, 'model', self._model)
        self._steps = Config.get_optional_value(cfg, 'steps', self._steps)
        self._opts = Config.get_optional_value(cfg, 'opts', self._opts)

    def yaml(self) -> dict:
        return {'metric': self._metric,'model': self._model,'steps': self._steps,'opts': self._opts.yaml()}

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
    NInputs:        int   = NecessaryField
    NReservoir:     int   = NecessaryField
    SpectralRadius: float = NecessaryField
    Sparsity:       int   = NecessaryField
    Noise:          float = NecessaryField
    LambdaR:        float = NecessaryField
    Inspect:        bool  = False
    RandomState:    int   = 0

    def load(self, cfg: dict) -> None:
        EsnConfigField.NInputs        = Config.get_necessary_value(cfg, 'n_inputs', EsnConfigField.NInputs)
        EsnConfigField.NReservoir     = Config.get_necessary_value(cfg, 'n_reservoir', EsnConfigField.NReservoir)
        EsnConfigField.SpectralRadius = Config.get_necessary_value(cfg, 'spectral_radius', EsnConfigField.SpectralRadius)
        EsnConfigField.Sparsity       = Config.get_necessary_value(cfg, 'sparsity', EsnConfigField.Sparsity)
        EsnConfigField.Noise          = Config.get_necessary_value(cfg, 'noise', EsnConfigField.Noise)
        EsnConfigField.LambdaR        = Config.get_necessary_value(cfg, 'lambda_r', EsnConfigField.LambdaR)
        EsnConfigField.Inspect        = Config.get_optional_value(cfg, 'inspect', EsnConfigField.Inspect)
        EsnConfigField.RandomState    = Config.get_optional_value(cfg, 'random_state', EsnConfigField.RandomState)

    def yaml(self) -> dict:
        return {'n_inputs': EsnConfigField.NInputs, 'n_reservoir': EsnConfigField.NReservoir, 'spectral_radius': EsnConfigField.SpectralRadius,
                'sparsity': EsnConfigField.Sparsity, 'noise': EsnConfigField.Noise, 'lambda_r': EsnConfigField.LambdaR, 'inspect': EsnConfigField.Inspect,
                'random_state': EsnConfigField.RandomState,}

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

class EvoLimitGenConfigField(ConfigSection):
    @property
    def Min(self) -> Union[int,float,None]: return self._min
    @property
    def Max(self) -> Union[int,float,None]: return self._max
    @property
    def IsInt(self) -> bool: return self._is_int

    def __init__(self) -> None:
        self._min:    Union[int,float,None] = None
        self._max:    Union[int,float,None] = None
        self._is_int: bool                  = False

    def load(self, cfg: dict) -> None:
        self._min    = Config.get_optional_value(cfg, 'min', self._min)
        self._max    = Config.get_optional_value(cfg, 'max', self._max)
        self._is_int = Config.get_optional_value(cfg, 'is_int', self._is_int)

    def yaml(self) -> dict:
        return {
            'min': self._min,'max': self._max,'is_int': self._is_int,
        }

class KVConfigField(ConfigSection):
    @property
    def Name(self) -> str: return self._name
    @property
    def Func(self) -> str: return self._func

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

class EvoSchemeConfigField(ConfigSection):
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
        self._max_gen_num:     int = NecessaryField
        self._population_size: int = NecessaryField
        self._rand_seed:       int = NecessaryField
        self._hromo_len:       int = NecessaryField
        self._hall_of_fame:    int = 0
        self._verbose:         bool = False

        self._fitness_weights: list = NecessaryField

        self._select: EvoSelectConfigField = EvoSelectConfigField()
        self._mate:   EvoMateConfigField   = EvoMateConfigField()
        self._mutate: EvoMutateConfigField = EvoMutateConfigField()

        self._limits: List[EvoLimitGenConfigField] = []
        self._metrics: List[EvoMetricConfigField] = []

    def load(self, cfg: dict) -> None:
        self._max_gen_num     = Config.get_necessary_value(cfg, 'max_gen_num', self._max_gen_num)
        self._population_size = Config.get_necessary_value(cfg, 'population_size', self._population_size)
        self._rand_seed       = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)
        self._hromo_len       = Config.get_necessary_value(cfg, 'hromo_len', self._hromo_len)
        self._hall_of_fame    = Config.get_optional_value(cfg, 'hall_of_fame', self._hall_of_fame)
        self._verbose         = Config.get_optional_value(cfg, 'verbose', self._verbose)

        self._fitness_weights = Config.get_necessary_value(cfg, 'fitness_weights', self._fitness_weights)

        self._select.load(Config.get_necessary_value(cfg, 'select', self._select))
        self._mate.load(Config.get_necessary_value(cfg, 'mate', self._mate))
        self._mutate.load(Config.get_necessary_value(cfg, 'mutate', self._mutate))

        self._limits = Config.get_optional_arr_value(cfg, 'limits', EvoLimitGenConfigField, self._limits)
        self._metrics = Config.get_optional_arr_value(cfg, 'metrics', EvoMetricConfigField, self._metrics)

    def yaml(self) -> dict:
        return {
            'max_gen_num': self._max_gen_num, 'population_size': self._population_size, 'rand_seed': self._rand_seed, 'hromo_len': self._hromo_len,
            'hall_of_fame': self._hall_of_fame,'fitness_weights': self._fitness_weights,'verbose': self._verbose,
            'select': self._select,'mate': self._mate,'mutate': self._mutate,
            'limits': _yaml_config_section_arr(self._limits),'metrics': _yaml_config_section_arr(self._metrics),
        }

class Scheme_1ConfigField(ConfigSection):
    @property
    def M(self) -> int: return self._m
    @property
    def C(self) -> int: return self._c
    @property
    def EvoSpec(self) -> EvoSchemeConfigField: return self._evo_spec

    def __init__(self) -> None:
        self._m:    int =  NecessaryField
        self._c:    int =  NecessaryField

        self._evo_spec: EvoSchemeConfigField = EvoSchemeConfigField()

    def load(self, cfg: dict) -> None:
        self._m = Config.get_necessary_value(cfg, 'm', self._m)
        self._c = Config.get_necessary_value(cfg, 'c', self._c)

        self._evo_spec.load(Config.get_necessary_value(cfg, 'evo_spec', self._evo_spec))

    def yaml(self) -> dict:
        return {'m': self._m, 'c': self._c,
                'evo_spec': self._evo_spec.yaml(),}

class EvoConfigField(ConfigSection):
    @property
    def Scheme_1(self) -> Scheme_1ConfigField: return self._scheme_1
    @property
    def Scheme_2(self) -> EvoSchemeConfigField: return self._scheme_2

    def __init__(self) -> None:
        self._scheme_1: Scheme_1ConfigField = Scheme_1ConfigField()
        self._scheme_2: EvoSchemeConfigField = EvoSchemeConfigField()

    def load(self, cfg: dict) -> None:
        self._scheme_1.load(Config.get_necessary_value(cfg, 'scheme_1', self._scheme_1))
        self._scheme_2.load(Config.get_necessary_value(cfg, 'scheme_2', self._scheme_2))

    def yaml(self) -> dict:
        return {'scheme_1': self._scheme_1.yaml(),'scheme_2': self._scheme_2.yaml(),}

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

class LorenzModelsPropConfigField(ConfigSection):
    @property
    def N(self): return self._n
    @property
    def Ro(self): return self._ro
    @property
    def RandSeed(self): return self._rand_seed
    @property
    def Dt(self): return self._dt

    def __init__(self) -> None:
        self._n:         int = 0
        self._ro:        float = 0
        self._rand_seed: int   = 0
        self._dt:        float = 0

    def load(self, cfg: dict) -> None:
        self._n = Config.get_necessary_value(cfg, 'n', self._n)
        self._ro = Config.get_necessary_value(cfg, 'ro', self._ro)
        self._rand_seed = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)
        self._dt = Config.get_necessary_value(cfg, 'dt', self._dt)

    def yaml(self) -> dict:
        return {'n': self._n, 'ro': self._ro, 'rand_seed': self._rand_seed, 'dt': self._dt,}

class ModelsConfigField(ConfigSection):
    Lorenz: LorenzModelsPropConfigField = LorenzModelsPropConfigField()

    def load(self, cfg: dict) -> None:
        ModelsConfigField.Lorenz.load(Config.get_necessary_value(cfg, 'lorenz', ModelsConfigField.Lorenz))

    def yaml(self) -> dict:
        return {'lorenz': ModelsConfigField.Lorenz.yaml(),}


class MultiStepTestPropConfigField(ConfigSection):
    @property
    def DataN(self): return self._data_n
    @property
    def StepN(self): return self._step_n

    def __init__(self) -> None:
        self._data_n: int = 0
        self._step_n: int = 0

    def load(self, cfg: dict) -> None:
        self._data_n = Config.get_optional_value(cfg, 'data_n', self._data_n)
        self._step_n = Config.get_optional_value(cfg, 'step_n', self._step_n)

    def yaml(self) -> dict:
        return {'data_n': self._data_n, 'step_n': self._step_n,}

class TestConfigField(ConfigSection):
    @property
    def MultiStep(self): return self._multi_step

    def __init__(self) -> None:
        self._multi_step: MultiStepTestPropConfigField = MultiStepTestPropConfigField()

    def load(self, cfg: dict) -> None:
        self._multi_step.load(Config.get_necessary_value(cfg, 'multi_step', self._multi_step))

    def yaml(self) -> dict:
        return {'multi_step': self._multi_step.yaml(),}

# Main config class

class Config:
    Logging:  LoggingConfigField  = LoggingConfigField()
    Evaluate: EvaluateConfigField = EvaluateConfigField()
    Test:     TestConfigField     = TestConfigField()
    Dump:     DumpConfigField     = DumpConfigField()
    Esn:      EsnConfigField      = EsnConfigField()
    Evo:      EvoConfigField      = EvoConfigField()
    Grid:     GridConfigField     = GridConfigField()
    Models:   ModelsConfigField   = ModelsConfigField()

    def load(cfg: dict, raise_if_necessary: bool=True) -> None:
        Config.__raise_if_necessary = raise_if_necessary

        # Load main sections
        Config.Logging.load(Config.get_optional_value(cfg, 'logging', Config.Logging))
        Config.Evaluate.load(Config.get_optional_value(cfg, 'evaluate', Config.Evaluate))
        Config.Test.load(Config.get_optional_value(cfg, 'test', Config.Test))
        Config.Dump.load(Config.get_optional_value(cfg, 'dumb', Config.Dump))
        Config.Esn.load(Config.get_necessary_value(cfg, 'esn', Config.Esn))
        Config.Evo.load(Config.get_necessary_value(cfg, 'evo', Config.Evo))
        Config.Grid.load(Config.get_necessary_value(cfg, 'grid', Config.Grid))
        Config.Models.load(Config.get_necessary_value(cfg, 'models', Config.Models))

        Config.__raise_if_necessary = True
        Config.__patched = True

    # Serialization config to yaml
    def yaml() -> dict:
        return {'logging': Config.Logging.yaml(),'evaluate': Config.Evaluate.yaml(), 'run': Config.Run.yaml(), 'test': Config.Test.yaml(), 'dumb': Config.Dump.yaml(), 'esn': Config.Esn.yaml(),
                'evo': Config.Evo.yaml(),'grid': Config.Grid.yaml(),'models': Config.Models.yaml(),}

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

def init(args):
    if args is not None and hasattr(args, 'disable_config'):
        return
    if args is None or not hasattr(args, 'config_path'):
        raise 'cant provide config path'
    if not hasattr(Config, '__patched'):
        Config.patch_from_file(args.config_path)
