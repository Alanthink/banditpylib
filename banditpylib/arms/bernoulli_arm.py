from typing import Union

import numpy as np

from .utils import StochasticArm


class BernoulliArm(StochasticArm):
  """Bernoulli arm

  Arm with rewards generated from a Bernoulli distribution.
  """
  def __init__(self, mu: float, name: str = None):
    """
    Args:
      mu: mean of rewards
      name: alias name
    """
    super().__init__(name)
    if (mu < 0) or (mu > 1):
      raise Exception('Mean of rewards is expected within [0, 1]. Got %.2f.' %
                      mu)
    self.__mu = mu

  def _name(self) -> str:
    return 'bernoulli_arm'

  @property
  def mean(self) -> float:
    return self.__mu

  def pull(self, pulls: int = None) -> Union[float, np.ndarray]:
    if pulls is None:
      return np.random.binomial(1, self.__mu, 1)[0]
    if pulls <= 0:
      raise ValueError('Number of pulls is expected at least 1. Got %d.' %
                       pulls)
    return np.random.binomial(1, self.__mu, pulls)
