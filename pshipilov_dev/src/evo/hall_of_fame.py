from typing import List, Tuple
import numpy as np

from deap import creator, tools, base

class DynoHallOfFame(tools.HallOfFame):
    def __init__(self, maxsize, similar=...):
        super().__init__(maxsize, similar)

    def update(self, population):
        if self.maxsize == 0 or len(population) == 0:
            return

        vals = [(np.dot(ind.fitness.values, ind.fitness.weights), ind) for ind in population]
        vals.sort(key=self._cmp)
        self_vals = [(np.dot(ind.fitness.values, ind.fitness.weights), ind) for ind in self]

        cnt = 0
        for val in vals:
            if cnt == self.maxsize:
                break
            cnt += 1


    def _cmp(self, l: Tuple[float, List], r: Tuple[float, List]) -> int:
        diff = l[0] - r[0]
        if np.fabs(diff) < 1e-6:
            return 0
        return l[0] - r[0]

