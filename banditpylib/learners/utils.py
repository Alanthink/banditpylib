"""
Abstract class for a learner.

Before each trial is run, a learner should be initialized with `reset`. The
parameters of `reset` may not be the same for different types of learners.
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
  def goal(self):
    """a string denoting the goal of the learner"""

  @abstractmethod
  def reset(self):
    """learner initialization

    This function should be called before the start of each trial of experiment.
    """
