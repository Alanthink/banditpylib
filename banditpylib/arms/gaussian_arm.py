import math
import numpy as np

from .utils import Arm


class GaussianArm(Arm):
  """Gaussian arm
  """

  def __init__(self, mu: float, var: float):
    """
    Args:
      mu: mean of the arm
      var: variance of the arm
    """
    self.__mu = mu
    if var < 0:
      raise Exception(
          'Variance of gaussian distribution %.2f is negative!' % var)
    self.__var = var
    # compute standard deviation
    self.__std = math.sqrt(var)

  @property
  def mean(self) -> float:
    """
    Returns:
      mean of the arm
    """
    return self.__mu

  @property
  def var(self) -> float:
    """
    Returns:
      variance of the arm
    """
    return self.__var

  def pull(self, pulls=1) -> np.ndarray or None:
    """Pulling the arm

    Args:
      pulls: number of pulls

    Returns:
      stochastic rewards. When number of pulls is less than 1, `None` is \
      returned.
    """
    if pulls < 1:
      return None
    return np.random.normal(self.__mu, self.__std, pulls)
