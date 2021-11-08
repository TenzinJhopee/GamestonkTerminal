from typing import Union

import numpy as np


class Normaliser:

    def __init__(self, x_min: Union[float, np.float], x_max: Union[float, np.float]):
        """Normalise inputs to lie in the range [-1, +1] or denormalise to natural range

        :param x_min: minimum value in feature series
        :param x_max: maximum value in feature series
        """
        self.x_min = x_min
        self.x_max = x_max

    def normalise(self, x: Union[float, np.float]) -> Union[float, np.float]:
        """Normalise input x

        :param x: input value for normalisation
        :return: normalised value
        """
        return (2 * (x - self.x_min) / (self.x_max - self.x_min)) - 1

    def denormalise(self, x: Union[float, np.float]) -> Union[float, np.float]:
        """Denormalise input x

        :param x: input value for denormalisation
        :return: denormalised value
        """
        return 0.5 * (x + 1) * (self.x_max - self.x_min) + self.x_min