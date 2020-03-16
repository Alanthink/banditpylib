"""
Abstract class for a learner.

Before a game runs, a learner should be initialized with `reset`.
"""
from abc import ABC, abstractmethod

__all__ = ['Learner']


class Learner(ABC):
  """abstract class for learners"""

  def __init__(self, pars):
    self.__name = pars['name'] if 'name' in pars else None

  @property
  def name(self):
    """learner name"""
    if self.__name:
      return self.__name
    return self._name

  @property
  @abstractmethod
  def _name(self):
    """learner default name"""

  @property
  @abstractmethod
  def goal(self):
    """goal of the learner"""

  @abstractmethod
  def reset(self, bandit, stop_cond):
    """learner initialization

    This function should be called before the start of the game.
    """
