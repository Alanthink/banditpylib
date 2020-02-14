from abc import ABC, abstractmethod

__all__ = ['Protocol']


class Protocol(ABC):
  """Abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  @abstractmethod
  def init(self):
    pass