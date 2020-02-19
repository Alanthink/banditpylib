import numpy as np
from .utils import Arm

__all__ = ['CorrelatedArm']


class CorrelatedArm(Arm):
  """Correlated arm"""

  def __init__(self, mu):
    self.__mu = mu

  @property
  def mean(self):
    return self.__mu

  def pull(self, pulls=1):
    """return a numpy array of stochastic rewards using true mean"""
    return self.__mu + np.random.normal(0, 1, pulls)
