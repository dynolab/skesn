from copy import copy
import numpy as np
import thequickmath.reduced_models.transition_to_turbulence as math_models

from typing import List, Union

import src.config as config

from .abstract import Model

class MoehlisModel(Model):
    def __init__(self,
        cfg: config.MoehlisModelConfig,
    ) -> None:
        super().__init__(9)
        self._cfg: config.MoehlisModelConfig = copy(cfg)
        self._model = math_models.MoehlisFaisstEckhardtModel(
            Re=self._cfg.Re,
            L_x=self._cfg.Lx,
            L_z=self._cfg.Lz,
        )
        self._rand = np.random.RandomState(self._cfg.RandSeed)

    def gen_data(
        self,
        n: int=None,
    ) -> np.ndarray:
        if n is None:
            n = self._cfg.N

        ret = np.zeros((n, self._dim))
        ret[0,:] = self._rand.random(self._dim)
        for i in range(1, n):
            ret[i,:] = self._model.f(ret[i-1])

        return ret

    def normalize(self,
        data: Union[list, np.ndarray],
    ) -> np.ndarray:
        if not self.valid_data(data):
            raise 'data is not valid for Moehlis model'

        ret: np.ndarray = None
        if isinstance(data, list):
            ret = np.array(data)
        elif isinstance(data, np.ndarray):
            ret = data.copy()

        n = len(data)
        for i in range(n):
            ret[i] = (ret[i] - [0,0,25]) / [30,30,30]

        return ret

    def denormalize(self,
        data: Union[list, np.ndarray],
    ) -> np.ndarray:
        if not self.valid_data(data):
            raise 'data is not valid for Moehlis model'

        if isinstance(data, list):
            data = np.array(data)

        return data*[[30],[30],[30]] + [[0],[0],[25]]
