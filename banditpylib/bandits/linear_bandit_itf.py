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
    """Total number of arms"""

  @abstractmethod
  def features(self) -> np.ndarray:
    """Feature vectors"""

  @abstractmethod
  def total_pulls(self) -> int:
    """Total number of pulls so far"""
