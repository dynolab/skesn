from copy import copy
import numpy as np

from typing import List, Union

import src.config as config

from .abstract import Model

class ChuiMoffattModel(Model):
    def __init__(self,
        cfg: config.ChuiMoffattModelConfig,
    ) -> None:
        self._cfg: config.ChuiMoffattModelConfig = copy(cfg)
        super().__init__(5)

    def gen_data(
        self,
        n: int=None,
    ) -> np.ndarray:
        if n is None:
            n = self._cfg.N

        ret = np.zeros((n, self._dim))

        rand = np.random.RandomState(self._cfg.RandSeed)
        ret[0,:] = rand.random(self._dim)
        for i in range(1, n):
            [x,y,z,u,v] = ret[i-1,:]
            ret[i,0] = x + self._cfg.Alpha*(-self._cfg.Eta*x+self._cfg.Omega*y*z)*self._cfg.Dt
            ret[i,1] = y + (-self._cfg.Eta*y+self._cfg.Omega*x*z)*self._cfg.Dt
            ret[i,2] = z + self._cfg.Kappa*(u-z-x*y)*self._cfg.Dt
            ret[i,3] = u + (-u+self._cfg.Xi*z-v*z)*self._cfg.Dt
            ret[i,4] = v + (-v+u*z)*self._cfg.Dt
        return ret

    def normalize(self,
        data: Union[list, np.ndarray],
    ) -> np.ndarray:
        if not self.valid_data(data):
            raise 'data is not valid for Chui-Moffatt model'

        # ret: np.ndarray = None
        # if isinstance(data, list):
        #     ret = np.array(data)
        # elif isinstance(data, np.ndarray):
        #     ret = data.copy()

        # n = len(data)
        # for i in range(n):
        #     ret[i] = (ret[i] - [0,0,25]) / [30,30,30]

        # return ret
        return data/[2,2,10,20,50]

    def denormalize(self,
        data: Union[list, np.ndarray],
    ) -> np.ndarray:
        if not self.valid_data(data):
            raise 'data is not valid for Chui-Moffatt model'

        # if isinstance(data, list):
        #     data = np.array(data)

        # return data*[[30],[30],[30]] + [[0],[0],[25]]
        return data*[[2],[2],[10],[20],[50]]

