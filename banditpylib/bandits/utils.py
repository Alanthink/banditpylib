from abc import ABC, abstractmethod


class Bandit(ABC):
  """
  Abstract class for the bandit environment.

  Before a game is run, one has to call :func:`reset` to initialize the
  environment. During each time step, :func:`context` will return the
  current state of the environment. :func:`feed` is used to pass the action to
  the environment for execution. The update of the state of is maintained by the
  bandit itself.
  """

  @property
  @abstractmethod
  def type(self):
    """str: type of the bandit environment"""

  @abstractmethod
  def reset(self):
    """Reset of the environment"""

  @property
  @abstractmethod
  def context(self):
    """Current state of the environment
    """

  @abstractmethod
  def _take_action(self, action):
    """
    Args:
      action (int or [(int, int), ]): if it is a list of two-tuples,
      in each two-tuple (A, B), A is the arm index and B is
      the number of actions to be taken; if it is an *int*, it is assumed the
      arm will be played only one time.

    Return:
      float or [float, ]: reward or a list of rewards
    """

  @abstractmethod
  def _update_context(self):
    pass

  def feed(self, action):
    """
    Args:
      actoin (int): action to take

    Return:
      (float, ): the first element denotes the reward
    """
    feedback = self._take_action(action)
    self._update_context()
    return feedback
