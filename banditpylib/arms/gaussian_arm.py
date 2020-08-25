import math
import numpy as np

from .utils import Arm


class GaussianArm(Arm):
  """Gaussian arm or Normal arm
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
    """mean of the arm"""
    return self.__mu

  @property
  def var(self) -> float:
    """variance of the arm"""
    return self.__var

  def pull(self, pulls=1) -> np.ndarray:
    if pulls < 1:
      raise Exception('Number of pulls %d is less than 1!' % pulls)
    return np.random.normal(self.__mu, self.__std, pulls)
