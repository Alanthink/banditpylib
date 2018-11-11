# -*- coding: utf-8 -*-
"""
Arms
"""

import numpy as np


class Arm:  # pylint: disable=too-few-public-methods
    """Base class for arm"""

    def __init__(self, mean):
        self.mean = mean


class BernoulliArm(Arm):  # pylint: disable=too-few-public-methods
    """Bernoulli arm"""

    def __init__(self, mean):
        Arm.__init__(self, mean)

    def pull(self):
        """return a stochastic reward"""
        return np.random.binomial(1, self.mean)


class GaussianArm(Arm):  # pylint: disable=too-few-public-methods
    """Gaussian arm"""

    def __init__(self, mu, sigma):
        Arm.__init__(self, mu)
        self.sigma = sigma

    def pull(self):
        """return a stochastic reward"""
        return np.random.normal(self.mean, self.sigma, 1)[0]
