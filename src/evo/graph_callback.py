from types import FunctionType
from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

class GraphCallbackModule(object):
    def __init__(self,
        graph_shape: Union[int, Tuple[int]],
    ) -> None:
        if isinstance(graph_shape, int):
            graph_shape = (graph_shape, 1)
        self._nrows = graph_shape[0]
        self._ncols = graph_shape[1]
        self._callbacks = []
        self._fig, self._axes = plt.subplots(nrows=self._nrows, ncols=self._ncols)

    def get_axes(self):
        return self._axes

    def get_fig(self):
        return self._fig

    def regist_callback(self,
        callback: FunctionType,
    ) -> None:
        self._callbacks.append(callback)

    def callback(self, **kvargs) -> None:
        if self._nrows == 1 and self._ncols == 1:
            self._callback_point(**kvargs)
        elif self._ncols == 1:
            self._callback_col(**kvargs)
        elif self._nrows == 1:
            self._callback_row(**kvargs)
        else:
            self._callback_grid(**kvargs)
        return

    def get_deap_callback(self):
        def wrap_callback(population, gen, **kvargs):
            self.callback(population=population, gen=gen, **kvargs)
        return wrap_callback

    def _callback_grid(self, **kvargs) -> None:
        for i in range(self._nrows):
            for j in range(self._ncols):
                for callback in self._callbacks:
                    callback(i, j, self._fig, self._axes, **kvargs)

    def _callback_col(self, **kvargs) -> None:
        for i in range(self._nrows):
            for callback in self._callbacks:
                    callback(i, 0, self._fig, self._axes, **kvargs)

    def _callback_row(self, **kvargs) -> None:
        for i in range(self._ncols):
            for callback in self._callbacks:
                    callback(0, i, self._fig, self._axes, **kvargs)

    def _callback_point(self, **kvargs) -> None:
        for callback in self._callbacks:
                callback(0, 0, self._fig, self._axes, **kvargs)
