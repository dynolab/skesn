import numpy as np

from typing import Union


class Model(object):
    def get_dim(self) -> int: pass

    def valid_data(self, data: Union[list, np.ndarray]) -> bool: pass

    def gen_data(self,  n: int=None) -> np.ndarray: pass

    def normalize(self, data: Union[list, np.ndarray]) -> np.ndarray: pass

    def denormalize(self, data: Union[list, np.ndarray]) -> np.ndarray: pass
