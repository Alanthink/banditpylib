from typing import Union

import numpy as np

from .utils import StochasticArm


class GaussianArm(StochasticArm):
  """Gaussian arm

  Arm with rewards generated from a Gaussian distribution.
  """
  def __init__(self, mu: float, std: float, name: str = None):
    """
    Args:
      mu: mean of rewards
      std: standard deviation of rewards
      name: alias name
    """
    super().__init__(name)
    if std <= 0:
      raise ValueError(
          'Standard deviation of rewards is expected > 0. Got %.2f.' % std)
    self.__mu = mu
    self.__std = std

  def _name(self) -> str:
    return 'gaussian_arm'

  @property
  def mean(self) -> float:
    return self.__mu

  @property
  def std(self) -> float:
    """Standard deviation of rewards"""
    return self.__std

  def pull(self, pulls: int = None) -> Union[float, np.ndarray]:
    if pulls is None:
      return np.random.normal(self.__mu, self.__std, 1)[0]
    if pulls <= 0:
      raise ValueError('Number of pulls is expected at least 1. Got %d.' %
                       pulls)
    return np.random.normal(self.__mu, self.__std, pulls)
