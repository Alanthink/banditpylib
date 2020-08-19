from abc import abstractmethod

import numpy as np
from .utils import Bandit


class LinearBanditItf(Bandit):
  """Linear bandit interface
  """

  def context(self):
    return None

  @abstractmethod
  def arm_num(self) -> int:
    """
    Return:
      total number of arms
    """

  @abstractmethod
  def features(self) -> np.ndarray:
    """
    Return:
      features of arms. First dimension: number of arms. Second dimension:
      dimension of the features.
    """

  @abstractmethod
  def total_pulls(self) -> int:
    """
    Return:
      total number of pulls so far
    """
