import math
from typing import Union

import numpy as np

from .utils import Arm


class GaussianArm(Arm):
  """Gaussian arm

  Arm with rewards generated from a Gaussian distribution.
  """
  def __init__(self, mu: float, var: float, name: str = None):
    """
    Args:
      mu: mean of rewards
      var: variance of rewards
      name: alias name
    """
    super().__init__(name)
    self.__mu = mu
    if var < 0:
      raise Exception('Variance of rewards %.2f is negative!' % var)
    self.__var = var
    # standard deviation
    self.__std = math.sqrt(var)

  def _name(self) -> str:
    """
    Returns:
      default arm name
    """
    return 'gaussian_arm'

  @property
  def mean(self) -> float:
    """mean of rewards"""
    return self.__mu

  @property
  def std(self) -> float:
    """standard deviation of rewards"""
    return self.__std

  @property
  def var(self) -> float:
    """variance of rewards"""
    return self.__var

  def pull(self, pulls: int = None) -> Union[float, np.ndarray]:
    """Pull the arm

    When pulls is None, a float number will be returned. Otherwise, a numpy
    array will be returned.

    Args:
      pulls: number of times to pull

    Returns:
      stochastic rewards
    """
    if pulls is None:
      return np.random.normal(self.__mu, self.__std, 1)[0]
    if pulls <= 0:
      raise ValueError('Number of pulls should be at least 1.')
    return np.random.normal(self.__mu, self.__std, pulls)
