from typing import Optional, Union

import numpy as np

from .utils import StochasticArm


class BernoulliArm(StochasticArm):
  """Bernoulli arm

  Arm with rewards generated from a Bernoulli distribution.

  :param float mu: mean of rewards
  :param Optional[str] name: alias name
  """
  def __init__(self, mu: float, name: Optional[str] = None):
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

  def pull(self, pulls: Optional[int] = None) -> Union[float, np.ndarray]:
    if pulls is None:
      return np.random.binomial(1, self.__mu, 1)[0]
    if pulls <= 0:
      raise ValueError('Number of pulls is expected at least 1. Got %d.' %
                       pulls)
    return np.random.binomial(1, self.__mu, pulls)
