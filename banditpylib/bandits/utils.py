"""
Abstract class for the bandit environment.

Before each trial is run, one has to call `init` function to initialize the
environment. During each time step, `context` will return the
current state of the environment. `feed` is used to pass the action to the
environment for execution. The update of the state of is maintained by the
bandit itself using `_update_context`.

The bandit environment should HIDE the sensitive information from outside.
"""
from abc import ABC, abstractmethod

__all__ = ['Bandit']


class Bandit(ABC):
  """abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    """name of the bandit environment"""

  @abstractmethod
  def init(self):
    """function to be called to initialize the environment"""

  # current state of the environment
  @property
  @abstractmethod
  def context(self):
    pass

  @abstractmethod
  def _take_action(self, action):
    """
    input:
      action: an integer or a list of two-tuples; if it is a list of two-tuples,
      in each two-tuple, the first item is the arm index; the second item is
      the number of actions to be taken; if it is an integer, it is assumed the
      arm index and will be pulled one time.
    return:
      reward or a list of rewards
    """

  @abstractmethod
  def _update_context(self):
    pass

  def feed(self, action):
    """
    input:
      action: an integer denoting which action to take
    return:
      feedback: a tuple and feedback[0] denotes the reward
    """
    feedback = self._take_action(action)
    self._update_context()
    return feedback