from io import IOBase
import json
from types import FunctionType

import yaml
import src.config as cfg

from src.models.abstract import Model
from src.models.lorenz import LorenzModel
from src.models.chui_moffatt import ChuiMoffattModel
from src.models.moehlis import MoehlisModel

from typing import List

import numpy as np

class EsnDataHolder:
    @property
    def Model(self) -> Model: return self._model

    @property
    def FitData(self) -> np.ndarray: return self._fit_data

    @property
    def ValidData(self) -> np.ndarray: return self._valid_data

    def __init__(self,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
    ) -> None:
        self._evaluate_cfg = evaluate_cfg

        if self._evaluate_cfg.Data is not None:
            return self._setup_data_by_file()
        elif self._evaluate_cfg.Model is not None:
            self._model: Model = _create_model_by_type(self._evaluate_cfg.Model)
            return self._setup_data_by_model()
        else:
            raise '"model" or "data" fields should be passed to evaluate config'

    def _setup_data(self,
        data: np.ndarray,
    ) -> None:
        split_n = self._evaluate_cfg.SplitN
        if split_n <= 0:
            split_n = data.shape[0] // 2

        if self._evaluate_cfg.FitStep > 0:
            fit_data = data[:split_n,]
            self._fit_data: np.ndarray = np.array([fit_data[i] for i in range(0, len(fit_data), self._evaluate_cfg.FitStep)])
        else:
            self._fit_data: np.ndarray = data[:split_n,]
        self._valid_data: np.ndarray = data[split_n:,:]

    def _setup_data_by_model(self) -> None:
        data = self._model.gen_data(self._evaluate_cfg.N)
        if self._evaluate_cfg.Normalize:
            data = self._model.normalize(data)
        self._setup_data(data)

    def _setup_data_by_file(self) -> None:
        data: np.ndarray = None
        with open(self._evaluate_cfg.Data, 'r') as f:
            if self._evaluate_cfg.Data.endswith('.json'):
                data = _get_simple_array_data_by_load_func(f, json.load)
            elif self._evaluate_cfg.Data.endswith('.yaml') or self._evaluate_cfg.Data.endswith('.yml'):
                data = _get_simple_array_data_by_load_func(f, yaml.full_load)
            else:
                raise 'unknown data files for model'
        return self._setup_data(data)

class EsnMultiDataHolder:
    @property
    def Models(self) -> List[Model]: return self._models

    @property
    def N(self) -> int: return self._n

    def FitDataByN(self,
        n: int,
    ) -> np.ndarray:
        return self._fit_datas[n]

    def ValidDataByN(self,
        n: int,
    ) -> np.ndarray:
        return self._valid_datas[n]

    def __init__(self,
        models: List[Model],
        split_n: int=0,
        fit_step: int=0,
        normalize: bool=False,
    ) -> None:
        self._models: List[Model] = [model for model in models]
        self._split_n: int = split_n
        self._fit_step: int = fit_step
        self._normalize: bool = normalize
        self._setup_data_by_model()

    def _setup_data_by_model(self) -> None:
        self._n = len(self._models)

        self._fit_datas: List[np.ndarray] = [None] * self._n
        self._valid_datas: List[np.ndarray] = [None] * self._n

        split_n = self._split_n
        if split_n <= 0:
            split_n = data.shape[0] // 2

        for i in range(self._n):
            data = self._models[i].gen_data()
            if self._normalize:
                data = self._models[i].normalize(data)

            if self._fit_step > 0:
                fit_data = data[:split_n,]
                self._fit_datas[i]: np.ndarray = np.array([fit_data[i] for i in range(0, len(fit_data), self._fit_step)])
            else:
                self._fit_datas[i] = data[:split_n,]
            self._valid_datas[i] = data[split_n:,:]

def _get_simple_array_data_by_load_func(f: IOBase, loader: FunctionType):
    raw_array = loader(f)
    return np.array(raw_array)

def _create_model_by_type(
    model_type: str,
) -> Model:
    model_type = model_type.lower()
    if model_type == 'lorenz':
        return LorenzModel(cfg.Config.Models.Lorenz)
    if model_type == 'chui_moffatt':
        return ChuiMoffattModel(cfg.Config.Models.ChuiMoffat)
    if model_type == 'moehlis':
        return MoehlisModel(cfg.Config.Models.Moehlis)
    raise f'unknown model - {model_type}'
