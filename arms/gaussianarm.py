import numpy as np

from .utils import Arm

__all__ = ['GaussianArm']


class GaussianArm(Arm):
  """Gaussian arm"""

  def __init__(self, mu, sigma):
    super().__init__()
    self.__mu = mu
    self.__sigma = sigma

  def pull(self, pulls=1):
    """return a numpy array of stochastic rewards"""
    return np.random.normal(self.__mu, self.__sigma, pulls)

  @property
  def mean(self):
    return self.__mu

  @property
  def std(self):
    return self.__sigma
