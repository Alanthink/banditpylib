from abc import abstractmethod

from .utils import Bandit


class OrdinaryBanditItf(Bandit):
  """Ordinary bandit interface
  """

  @abstractmethod
  def arm_num(self) -> int:
    """Total number of arms"""

  @abstractmethod
  def total_pulls(self) -> int:
    """Total number of pulls executed"""
