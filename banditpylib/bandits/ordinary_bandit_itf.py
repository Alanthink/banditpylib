from abc import abstractmethod

from .utils import Bandit


class OrdinaryBanditItf(Bandit):
  """Ordinary bandit interface"""

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
  def total_pulls(self) -> int:
    """
    Returns:
      total number of pulls
    """
