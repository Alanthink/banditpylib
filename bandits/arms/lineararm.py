import numpy as np
from .utils import Arm

__all__ = ['LinearArm']


class LinearArm(Arm):
  """linear arm"""

  def __init__(self, feature, theta, var=1):
    self.__feature = feature
    self.__mu = np.dot(self.__feature, theta)
    self.__var = var

  @property
  def mean(self):
    return self.__mu

  @property
  def feature(self):
    return self.__feature

  def pull(self, pulls=1):
    """return a numpy array of stochastic rewards using true mean"""
    return self.__mu + np.random.normal(0, self.__var, pulls)
