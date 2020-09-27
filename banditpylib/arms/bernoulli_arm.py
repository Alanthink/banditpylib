import numpy as np

from .utils import Arm


class BernoulliArm(Arm):
  """Bernoulli arm
  """

  def __init__(self, mean: float):
    """
    Args:
      mean: real mean of the arm
    """
    if mean < 0:
      raise Exception('Mean of Bernoulli arm %.2f is less than 0!' % mean)
    if mean > 1:
      raise Exception('Mean of Bernoulli arm %.2f is greater than 1!' % mean)
    self.__mean = mean

  @property
  def mean(self) -> float:
    """real mean of the arm"""
    return self.__mean

  def pull(self, pulls: int=1) -> np.ndarray or None:
    """Pulling the arm

    Args:
      pulls: number of pulls

    Returns:
      stochastic rewards. When number of pulls is less than 1, `None` is \
      returned.
    """
    if pulls < 1:
      return None
    return np.random.binomial(1, self.__mean, pulls)
