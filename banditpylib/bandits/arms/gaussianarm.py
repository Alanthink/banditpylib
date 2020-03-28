import math
import numpy as np

from .utils import Arm


class GaussianArm(Arm):
  """Class for Gaussian arm"""

  def __init__(self, mu, var):
    """
    Args:
      mu (float): mean
      var (float): variance
    """
    self.__mu = mu
    self.__var = var
    self.__std = math.sqrt(var)

  @property
  def mean(self):
    return self.__mu

  @property
  def var(self):
    """Variance of the arm

    Return:
      float: variance of the arm
    """
    return self.__var

  def pull(self, pulls=1):
    return np.random.normal(self.__mu, self.__std, pulls)
