"""
Abstract class for a learner.

Before each trial is run, a learner must be initialized with `init`. The
parameters of `init` may not be the same for different types of learners.
However, the first argument should always be the bandit instance since during
the initialization, a learner may want to ask the bandit for some information.
`reward` should return a function of the learner such that the protocol knows
which function of the learner to call to ask for the empirical rewards.
`regret_def` returns a string will be used by the protocol to ask the bandit for
the correct regret.
"""
from abc import ABC, abstractmethod
from absl import logging

__all__ = ['Learner']


class Learner(ABC):
  """abstract class for learners"""

  def __init__(self, pars):
    self._name = pars['name'] if 'name' in pars else None

  @property
  def name(self):
    """name used to plot the final figure"""
    if not self._name:
      logging.fatal('Learner name is not defined!')
    return self._name

  @property
  def goal(self):
    """a string denoting the goal of the learner"""

  @property
  @abstractmethod
  def rewards_def(self):
    """a function for returning of the empirical reward"""

  @property
  @abstractmethod
  def regret_def(self):
    """a string defining the regret function"""
