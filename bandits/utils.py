from abc import ABC, abstractmethod

__all__ = ['Bandit']


class Bandit(ABC):
  """abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  @abstractmethod
  def init(self):
    pass

  # current state of the environment
  @property
  @abstractmethod
  def context(self):
    pass

  @abstractmethod
  def _take_action(self, action):
    """
    Input:
      action: an integer or a list of two-tuples; if it is a list of two-tuples,
      in each two-tuple, the first item is the arm index; the second item is
      the number of actions to be taken; if it is an integer, it is assumed the
      arm index and will be pulled one time.
    Output:
      reward or a list of rewards
    """

  @abstractmethod
  def _update_context(self):
    pass

  def feed(self, action):
    """
    Output:
      feedback: a tuple and feedback[0] denotes the reward
    """
    feedback = self._take_action(action)
    self._update_context()
    return feedback
