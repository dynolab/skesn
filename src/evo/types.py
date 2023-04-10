from typing import Iterable, List
from deap import base


class Fitness(base.Fitness):
    def __init__(self, values: Iterable[float]=()):
        super().__init__(values)

    @staticmethod
    def patch_weights(weights: Iterable[float]):
        Fitness.weights = weights

class Individual(list):
    def __init__(self, iterable):
        self.fitness = Fitness()
        super().__init__(iterable)
