import numpy as np

from typing import List, Union


class Model(object):
    def __init__(self,
        dim: int,
    ) -> None:
        self._dim:int = dim

    # Abstract methods

    def gen_data(self,  n: int=None) -> np.ndarray: pass

    def normalize(self, data: Union[list, np.ndarray]) -> np.ndarray: pass

    def denormalize(self, data: Union[list, np.ndarray]) -> np.ndarray: pass

    # Public methods

    def get_dim(self) -> int: return self._dim

    def valid_data(self,
        data: Union[List, np.ndarray],
    ) -> bool:
        if isinstance(data, list):
            return self._valid_list(data)
        elif isinstance(data, np.ndarray):
            return self._valid_ndarray(data)
        return False

    # Private methods

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
