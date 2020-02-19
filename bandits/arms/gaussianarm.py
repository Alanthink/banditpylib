import numpy as np

from .utils import Arm

__all__ = ['GaussianArm']


class GaussianArm(Arm):
  """Gaussian arm"""

  def __init__(self, mu, sigma):
    self.__mu = mu
    self.__sigma = sigma

  @property
  def mean(self):
    return self.__mu

  @property
  def std(self):
    return self.__sigma

  def pull(self, pulls=1):
    """return a numpy array of stochastic rewards"""
    return np.random.normal(self.__mu, self.__sigma, pulls)
