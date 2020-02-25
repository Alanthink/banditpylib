"""
Abstract learner
"""
from abc import ABC, abstractmethod
from absl import logging

__all__ = ['Learner']


class Learner(ABC):
  """Abstract class for learners"""

  def __init__(self, pars):
    self._name = pars['name'] if 'name' in pars else None

  # learner goal
  @property
  @abstractmethod
  def goal(self):
    pass

  # learner name
  @property
  def name(self):
    if not self._name:
      logging.fatal('Learner name is not defined!')
    return self._name
