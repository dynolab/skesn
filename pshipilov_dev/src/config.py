from typing import Any

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

class LoggingConfigField(ConfigSection):
    Level:   str  = 'info'
    File:    str  = 'logs/log'
    Console: bool = True
    Enable:  bool = True

    def load(self, cfg: dict) -> None:
        LoggingConfigField.Level   = Config.get_optional_value(cfg, 'level', LoggingConfigField.Level)
        LoggingConfigField.File    = Config.get_optional_value(cfg, 'file', LoggingConfigField.File)
        LoggingConfigField.Console = Config.get_optional_value(cfg, 'console', LoggingConfigField.Console)
        LoggingConfigField.Enable  = Config.get_optional_value(cfg, 'enable', LoggingConfigField.Enable)

    def yaml(self) -> dict:
        return {'level': LoggingConfigField.Level,'file': LoggingConfigField.File,'console': LoggingConfigField.Console,
                'enable': LoggingConfigField.Enable,}

class DumpConfigField(ConfigSection):
    Enable: bool = True
    User:   str  = 'auto'
    Dir:    str  = 'dumps'
    Title:  str  = ''

    def load(self, cfg: dict) -> None:
        DumpConfigField.Enable = Config.get_optional_value(cfg, 'enable', DumpConfigField.Enable)
        DumpConfigField.User   = Config.get_optional_value(cfg, 'user', DumpConfigField.User)
        DumpConfigField.Dir    = Config.get_optional_value(cfg, 'dir', DumpConfigField.Dir)
        DumpConfigField.Title  = Config.get_optional_value(cfg, 'title', DumpConfigField.Title)

    def yaml(self) -> dict:
        return {'title': DumpConfigField.Title, 'enable': DumpConfigField.Enable, 'user': DumpConfigField.User,'dirname': DumpConfigField.Dir,}

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

class CommonEvoPropConfigField(ConfigSection):
    @property
    def MaxGenNum(self): return self._max_gen_num
    @property
    def PopulationSize(self): return self._population_size
    @property
    def RandSeed(self): return self._rand_seed

    @property
    def CrossoverP(self): return self._crossover_p
    @property
    def MutationP(self): return self._mutation_p

    @property
    def Scoring(self): return self._scoring
    @property
    def ValidMultiN(self): return self._valid_multi_n

    @property
    def Select(self): return self._select
    @property
    def Crossing(self): return self._crossing
    @property
    def Mutation(self): return self._mutation

    @property
    def MutationIndpb(self): return self._mutation_indpb


    @property
    def Verbose(self): return self._verbose


    def __init__(self) -> None:
        self._max_gen_num:     int = NecessaryField
        self._population_size: int = NecessaryField
        self._rand_seed:       int = NecessaryField

        self._crossover_p:     float = NecessaryField
        self._mutation_p:      float = NecessaryField

        self._select:          str = ''
        self._crossing:        str = ''
        self._mutation:        str = ''

        self._mutation_indpb:  float = -1

        self._scoring:         str = NecessaryField
        self._valid_multi_n:   int = -1

        self._verbose:         bool = False

    def load(self, cfg: dict) -> None:
        self._max_gen_num     = Config.get_necessary_value(cfg, 'max_gen_num', self._max_gen_num)
        self._population_size = Config.get_necessary_value(cfg, 'population_size', self._population_size)
        self._rand_seed       = Config.get_necessary_value(cfg, 'rand_seed', self._rand_seed)

        self._crossover_p     = Config.get_necessary_value(cfg, 'crossover_p', self._crossover_p)
        self._mutation_p      = Config.get_necessary_value(cfg, 'mutation_p', self._mutation_p)

        self._verbose         = Config.get_necessary_value(cfg, 'verbose', self._verbose)

        self._scoring         = Config.get_necessary_value(cfg, 'scoring', self._scoring)
        self._valid_multi_n   = Config.get_optional_value(cfg, 'valid_multi_n', self._valid_multi_n)

        self._select          = Config.get_optional_value(cfg, 'select', self._select)
        self._crossing        = Config.get_optional_value(cfg, 'crossing', self._crossing)
        self._mutation        = Config.get_optional_value(cfg, 'mutation', self._mutation)

        self._mutation_indpb  = Config.get_optional_value(cfg, 'mutation_indpb', self._mutation_indpb)

    def yaml(self) -> dict:
        return {'max_gen_num': self._max_gen_num, 'population_size': self._population_size, 'rand_seed': self._rand_seed,
                'crossover_p': self._crossover_p, 'mutation_p': self._mutation_p, 'verbose': self._verbose,'scoring': self._scoring,'valid_multi_n': self._valid_multi_n,
                'select': self._select, 'crossing': self._crossing, 'mutation': self._mutation,'mutation_indpb': self._mutation_indpb}

class Scheme_1ConfigField(ConfigSection):
    M:         int                      = NecessaryField
    C:         int                      = NecessaryField
    Weights:   list                     = NecessaryField
    Common:    CommonEvoPropConfigField = CommonEvoPropConfigField()

    def load(self, cfg: dict) -> None:
        Scheme_1ConfigField.M           = Config.get_necessary_value(cfg, 'm', Scheme_1ConfigField.M)
        Scheme_1ConfigField.C           = Config.get_necessary_value(cfg, 'c', Scheme_1ConfigField.C)
        Scheme_1ConfigField.Weights     = Config.get_necessary_value(cfg, 'weights', Scheme_1ConfigField.Weights)
    
        Scheme_1ConfigField.Common.load(Config.get_necessary_value(cfg, 'common', Scheme_1ConfigField.Common))

    def yaml(self) -> dict:
        return {'M': Scheme_1ConfigField.M, 'C': Scheme_1ConfigField.C, 'weights': Scheme_1ConfigField.Weights,
                'common': Scheme_1ConfigField.Common.yaml(),}

class LimitsParamPropConfigField(ConfigSection):
    @property
    def Min(self): return self._min
    @property
    def Max(self): return self._max
    @property
    def IsInt(self): return self._is_int

    def __init__(self) -> None:
        self._min:    float = 0 # NecessaryField
        self._max:    float = 0 # NecessaryField
        self._is_int: bool  = False

    def load(self, cfg: dict) -> None:
        self._min = Config.get_optional_value(cfg, 'min', self._min)
        self._max = Config.get_optional_value(cfg, 'max', self._max)
        self._is_int = Config.get_optional_value(cfg, 'is_int', self._is_int)

    def yaml(self) -> dict:
        return {'min': self._min,'max': self._max,'is_int': self._is_int,}

class LimitsScheme_2PropConfigField(ConfigSection):
    NReservoir:     LimitsParamPropConfigField = LimitsParamPropConfigField()
    SpectralRadius: LimitsParamPropConfigField = LimitsParamPropConfigField()
    Sparsity:       LimitsParamPropConfigField = LimitsParamPropConfigField()
    Noise:          LimitsParamPropConfigField = LimitsParamPropConfigField()
    LambdaR:        LimitsParamPropConfigField = LimitsParamPropConfigField()

    def load(self, cfg: dict) -> None:
        LimitsScheme_2PropConfigField.NReservoir.load(Config.get_optional_value(cfg, 'n_reservoir', {}))
        LimitsScheme_2PropConfigField.SpectralRadius.load(Config.get_optional_value(cfg, 'spectral_radius', {}))
        LimitsScheme_2PropConfigField.Sparsity.load(Config.get_optional_value(cfg, 'sparsity', {}))
        LimitsScheme_2PropConfigField.Noise.load(Config.get_optional_value(cfg, 'noise', {}))
        LimitsScheme_2PropConfigField.LambdaR.load(Config.get_optional_value(cfg, 'lambda_r', {}))

    def yaml(self) -> dict:
        return {'n_reservoir': LimitsScheme_2PropConfigField.NReservoir.yaml(),'spectral_radius': LimitsScheme_2PropConfigField.SpectralRadius.yaml(),'sparsity': LimitsScheme_2PropConfigField.Sparsity.yaml(),
            'noise': LimitsScheme_2PropConfigField.Noise.yaml(),'lambda_r': LimitsScheme_2PropConfigField.LambdaR.yaml(),}

class Scheme_2ConfigField(ConfigSection):
    Weights:   list                          = NecessaryField
    Common:    CommonEvoPropConfigField      = CommonEvoPropConfigField()
    Limits:    LimitsScheme_2PropConfigField = LimitsScheme_2PropConfigField()

    def load(self, cfg: dict) -> None:
        Scheme_2ConfigField.Weights     = Config.get_necessary_value(cfg, 'weights', Scheme_2ConfigField.Weights)
        Scheme_2ConfigField.Common.load(Config.get_necessary_value(cfg, 'common', Scheme_2ConfigField.Common))
        Scheme_2ConfigField.Limits.load(Config.get_necessary_value(cfg, 'limits', Scheme_2ConfigField.Limits))

    def yaml(self) -> dict:
        return {'weights': Scheme_2ConfigField.Weights,'common': Scheme_2ConfigField.Common.yaml(),'limits': Scheme_2ConfigField.Limits.yaml(),}

class EvoConfigField(ConfigSection):
    Scheme_1: Scheme_1ConfigField = Scheme_1ConfigField()
    Scheme_2: Scheme_2ConfigField = Scheme_2ConfigField()

    def load(self, cfg: dict) -> None:
        EvoConfigField.Scheme_1.load(Config.get_necessary_value(cfg, 'scheme_1', EvoConfigField.Scheme_1))
        EvoConfigField.Scheme_2.load(Config.get_necessary_value(cfg, 'scheme_2', EvoConfigField.Scheme_2))

    def yaml(self) -> dict:
        return {'scheme_1': EvoConfigField.Scheme_1.yaml(),'scheme_2': EvoConfigField.Scheme_2.yaml(),}

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
    Scoring:     str                    = NecessaryField
    ValidMultiN: int                    = -1
    Verbose:     bool                   = False
    ParamsSet:   ParamsSetGridPropField = ParamsSetGridPropField()

    def load(self, cfg: dict) -> None:
        GridConfigField.Scoring = Config.get_necessary_value(cfg, 'scoring', GridConfigField.Scoring)
        GridConfigField.Verbose = Config.get_optional_value(cfg, 'verbose', GridConfigField.Verbose)
        GridConfigField.ValidMultiN = Config.get_optional_value(cfg, 'valid_multi_n', GridConfigField.ValidMultiN)
        GridConfigField.ParamsSet.load(Config.get_necessary_value(cfg, 'params_set', GridConfigField.ParamsSet))

        # Other validation
        if GridConfigField.Scoring == 'valid_multi' \
            and GridConfigField.ValidMultiN < 2:
            raise(Exception('If "grid.scoring" = "valid_multi" then had to bind "grid.valid_multi_n" > 1'))

    def yaml(self) -> dict:
        return {'params_set': GridConfigField.ParamsSet.yaml(),'valid_multi_n': GridConfigField.ValidMultiN, 'verbose': GridConfigField.Verbose, 'scoring': GridConfigField.Scoring,}
        # return {'valid_multi_n': GridConfigField.ValidMultiN, 'verbose': GridConfigField.Verbose, 'scoring': GridConfigField.Scoring,}

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
    Logging: LoggingConfigField = LoggingConfigField()
    Test:    TestConfigField    = TestConfigField()
    Dump:    DumpConfigField    = DumpConfigField()
    Esn:     EsnConfigField     = EsnConfigField()
    Evo:     EvoConfigField     = EvoConfigField()
    Grid:    GridConfigField    = GridConfigField()
    Models:  ModelsConfigField  = ModelsConfigField()

    def load(cfg: dict, raise_if_necessary: bool=True) -> None:
        Config.__raise_if_necessary = raise_if_necessary

        # Load main sections
        Config.Logging.load(Config.get_optional_value(cfg, 'logging', Config.Logging))
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
        return {'logging': Config.Logging.yaml(), 'run': Config.Run.yaml(), 'test': Config.Test.yaml(), 'dumb': Config.Dump.yaml(), 'esn': Config.Esn.yaml(),
                'evo': Config.Evo.yaml(),'grid': Config.Grid.yaml(),'models': Config.Models.yaml(),}

    # Internal property for control raising exeption if field is not provided 
    __raise_if_necessary: bool = True

    # Internal helper func for patching

    def get_optional_value(cfg: dict, key: str, default: Any=None) -> Any:
        if key not in cfg:
            return default
        return cfg[key]

    def get_necessary_value(cfg: dict, key: str, default: Any=None) -> Any:
        if key not in cfg:
            if not Config.__raise_if_necessary:
                return default
            raise Exception(f'Failed config load, key \"{key}\" is necessary')
        return cfg[key]

    # Pathcing functions

    def patch_from_file(filename: str, raise_if_necessary: bool=True) -> None:
        with open(filename, "r") as config_stream:
            Config.load(yaml.safe_load(config_stream), raise_if_necessary)

    def patch_from_dict(cfg: dict, raise_if_necessary: bool=True) -> None:
        Config.load(cfg, raise_if_necessary)

if not hasattr(Config, '__patched'):
    Config.patch_from_file('pshipilov_dev/configs/config.yaml')
