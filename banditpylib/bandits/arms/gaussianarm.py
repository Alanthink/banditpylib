import math
import numpy as np

from .utils import Arm

__all__ = ['GaussianArm']


class GaussianArm(Arm):
  """gaussian arm"""

  def __init__(self, mu, var):
    self.__mu = mu
    self.__var = var
    self.__std = math.sqrt(var)

  @property
  def mean(self):
    return self.__mu

  @property
  def var(self):
    return self.__var

  def pull(self, pulls=1):
    """return a numpy array of stochastic rewards"""
    return np.random.normal(self.__mu, self.__std, pulls)
