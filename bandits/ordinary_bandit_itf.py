from abc import abstractmethod

from .utils import Bandit


class OrdinaryBanditItf(Bandit):
  """Ordinary bandit interface
  """

  @abstractmethod
  def arm_num(self) -> int:
    """
    Return:
      total number of arms
    """

  @abstractmethod
  def total_pulls(self) -> int:
    """
    Return:
      total number of pulls executed
    """
