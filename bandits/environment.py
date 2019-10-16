"""
Bandit
"""

from abc import ABC, abstractmethod


class Environment(ABC):
  """Abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  @property
  @abstractmethod
  def context(self):
    pass

  @abstractmethod
  def init(self):
    pass

  @abstractmethod
  def _best_pull(self, context):
    pass

  @abstractmethod
  def pull(self, context, action):
    pass

  @abstractmethod
  def regret(self, rewards):
    pass
