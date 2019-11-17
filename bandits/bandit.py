"""
Bandit
"""

from abc import ABC, abstractmethod


class Bandit(ABC):
  """Abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  # current state of the environment
  @property
  @abstractmethod
  def context(self):
    pass

  # full state of the environment (can not be fetched from outside)
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
    """
    Output:
      feedback: a tuple and feedback[0] denotes the reward
    """
    feedback = self._take_action(action)
    self._update_context()
    return feedback
