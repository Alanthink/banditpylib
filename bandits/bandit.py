"""
Bandit
"""

from abc import ABC, abstractmethod


class BanditEnvironment(ABC):
  """Abstract bandit environment"""

  @abstractmethod
  def init(self):
    pass

  @abstractmethod
  def context(self):
    pass

  @abstractmethod
  def pull(self, context, action):
    pass

  @abstractmethod
  def regret(self, rewards):
    pass

  @abstractmethod
  def type(self):
    pass
