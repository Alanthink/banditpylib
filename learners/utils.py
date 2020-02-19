"""
Abstract learner
"""
from abc import ABC, abstractmethod

__all__ = ['Learner']


class Learner(ABC):
  """Abstract class for learners"""

  # learner goal
  @property
  @abstractmethod
  def goal(self):
    pass

  # learner name
  @property
  @abstractmethod
  def name(self):
    pass
