from abc import abstractmethod

from typing import List

import numpy as np
from .utils import Bandit


class LinearBanditItf(Bandit):
  """Finite-armed linear bandit interface"""

  def context(self):
    """
    Returns:
      current state of the bandit environment
    """
    return None

  @abstractmethod
  def arm_num(self) -> int:
    """
    Returns:
      total number of arms
    """

  @abstractmethod
  def features(self) -> List[np.ndarray]:
    """
    Returns:
      feature vectors
    """

  @abstractmethod
  def total_pulls(self) -> int:
    """
    Returns:
      total number of pulls so far
    """
