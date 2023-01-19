from typing import List
from src.models.abstract import Model

import numpy as np

class EsnDataHolder:
    @property
    def Model(self) -> Model: return self._model

    @property
    def FitData(self) -> np.ndarray: return self._fit_data

    @property
    def ValidData(self) -> np.ndarray: return self._valid_data

    def __init__(self,
        model: Model,
        split_n: int=0,
        fit_step: int=0,
        normalize: bool=False,
    ) -> None:
        self._model: Model = model
        self._split_n: int = split_n
        self._fit_step: int = fit_step
        self._normalize: bool = normalize
        self._setup_data_by_model()

    def _setup_data_by_model(self) -> None:
        data = self._model.gen_data()
        if self._normalize:
            data = self._model.normalize(data)

        split_n = self._split_n
        if split_n <= 0:
            split_n = data.shape[0] // 2

        if self._fit_step > 0:
            fit_data = data[:split_n,]
            self._fit_data: np.ndarray = np.array([fit_data[i] for i in range(0, len(fit_data), self._fit_step)])
        else:
            self._fit_data: np.ndarray = data[:split_n,]
        self._valid_data: np.ndarray = data[split_n:,:]

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
