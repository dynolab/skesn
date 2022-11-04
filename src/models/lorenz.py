import numpy as np

from typing import List, Union

import src.config as config

from .abstract import Model

class LorenzModel(Model):
    def __init__(self,
        cfg: config.LorenzModelConfig,
    ) -> None:
        self._cfg: config.LorenzModelConfig = cfg
        self._dim: int = 3

    def get_dim(self) -> int:
        return self._dim

    def gen_data(
        self,
        n: int=None,
    ) -> np.ndarray:
        if n is None:
            n = self._cfg.N

        ret = np.zeros((n, self._dim))

        rand = np.random.RandomState(self._cfg.RandSeed)
        ret[0,:] = rand.random(3)
        for i in range(1, n):
            [x,y,z] = ret[i-1,:]
            ret[i,0] = x + 10*(y-x)*self._cfg.Dt
            ret[i,1] = y + (x*(self._cfg.Ro-z)-y)*self._cfg.Dt
            ret[i,2] = z + (x*y-8*z/3.)*self._cfg.Dt
        return ret

    def valid_data(self,
        data: Union[List, np.ndarray],
    ) -> bool:
        if isinstance(data, list):
            return self._valid_list(data)
        elif isinstance(data, np.ndarray):
            return self._valid_ndarray(data)
        return False

    def normalize(self,
        data: Union[list, np.ndarray],
    ) -> np.ndarray:
        if not self.valid_data(data):
            raise 'data is not valid for Lorenz model'

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
            raise 'data is not valid for Lorenz model'

        if isinstance(data, list):
            data = np.array(data)

        return data*[[30],[30],[30]] + [[0],[0],[25]]

    def _valid_ndarray(self,
        data: np.ndarray,
    ) -> bool:
        if len(data.shape) != 2:
            return False

        if data.shape[1] == self._dim:
            return True

        return False

    def _valid_list(self,
        data: List,
    ) -> bool:
        if len(data) == 0:
            return True

        for line in data:
            if not (isinstance(line, list) and len(line) == self._dim or\
                isinstance(line, np.ndarray) and line.shape == self._dim,):
                return False
        return True
