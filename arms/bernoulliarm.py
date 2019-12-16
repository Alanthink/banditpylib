import numpy as np

from .utils import Arm

__all__ = ['BernoulliArm']


class BernoulliArm(Arm):
  """Bernoulli arm"""

  def __init__(self, mean):
    self.__mean = mean

  @property
  def mean(self):
    return self.__mean

  def pull(self, pulls=1):
    """return a numpy array of stochastic rewards"""
    return np.random.binomial(1, self.__mean, pulls)
