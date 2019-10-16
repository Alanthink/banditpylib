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

  @property
  @abstractmethod
  def _oracle_context(self):
    pass

  @abstractmethod
  def init(self):
    pass

  @abstractmethod
  def _update_context(self):
    pass

  @abstractmethod
  def _take_action(self, action):
    pass

  @abstractmethod
  def regret(self, rewards):
    pass

  def feed(self, action):
    feedback = self._take_action(action)
    self._update_context()
    return feedback
