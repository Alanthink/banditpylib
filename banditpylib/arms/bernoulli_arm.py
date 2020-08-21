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
    """
    Return:
      real mean of the arm
    """
    return self.__mean

  def pull(self, pulls=1) -> np.ndarray:
    if pulls < 1:
      raise Exception('Number of pulls %d is less than 1!' % pulls)
    return np.random.binomial(1, self.__mean, pulls)
