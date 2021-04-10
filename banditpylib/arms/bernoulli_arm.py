import numpy as np

from .utils import Arm


class BernoulliArm(Arm):
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
    if mu < 0:
      raise Exception('Mean of rewards %.2f is less than 0!' % mu)
    if mu > 1:
      raise Exception('Mean of rewards %.2f is greater than 1!' % mu)
    self.__mu = mu

  def _name(self) -> str:
    """
    Returns:
      default arm name
    """
    return 'bernoulli_arm'

  @property
  def mean(self) -> float:
    """mean of rewards"""
    return self.__mu

  def pull(self, pulls: int = 1) -> np.ndarray:
    """Pull the arm

    Args:
      pulls: number of times to pull

    Returns:
      stochastic rewards
    """
    return np.random.binomial(1, self.__mu, pulls)
