"""
Arms
"""

from abc import ABC, abstractmethod

import numpy as np


class Arm(ABC):
  """Base class for an arm in the classic bandit model"""

  @abstractmethod
  def pull(self):
    pass

  @abstractmethod
  def mean(self):
    pass


class BernoulliArm(Arm):
  """Bernoulli arm"""

  def __init__(self, mean):
    super().__init__()
    self._mean = mean

  def pull(self):
    """return a stochastic reward"""
    return np.random.binomial(1, self._mean)

  @property
  def mean(self):
    return self._mean



class GaussianArm(Arm):
  """Gaussian arm"""

  def __init__(self, mu, sigma):
    super().__init__()
    self._mu = mu
    self._sigma = sigma

  def pull(self):
    """return a stochastic reward"""
    return np.random.normal(self._mu, self._sigma, 1)[0]

  @property
  def mean(self):
    return self._mu
