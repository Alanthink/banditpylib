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
    self.__mean = mean

  def pull(self):
    """return a stochastic reward"""
    return np.random.binomial(1, self.__mean)

  @property
  def mean(self):
    return self.__mean



class GaussianArm(Arm):
  """Gaussian arm"""

  def __init__(self, mu, sigma):
    super().__init__()
    self.__mu = mu
    self.__sigma = sigma

  def pull(self):
    """return a stochastic reward"""
    return np.random.normal(self.__mu, self.__sigma, 1)[0]

  @property
  def mean(self):
    return self.__mu
