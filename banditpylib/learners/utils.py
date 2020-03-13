"""
Abstract class for a learner.

Before a game runs, a learner should be initialized with `reset`.
"""
from abc import ABC, abstractmethod

__all__ = ['Learner']


class Learner(ABC):
  """abstract class for learners"""

  def __init__(self, pars):
    self._name = pars['name'] if 'name' in pars else None

  @property
  @abstractmethod
  def name(self):
    """learner name"""

  @property
  @abstractmethod
  def goal(self):
    """goal of the learner"""

  @abstractmethod
  def reset(self, bandit, stop_cond):
    """learner initialization

    This function should be called before the start of the game.
    """
