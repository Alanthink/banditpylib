from abc import ABC, abstractmethod


class Bandit(ABC):
  """Class for bandit environment

  Before a game is run, one has to call :func:`reset` to initialize the
  environment. During each time step, :func:`context` will return the
  current state of the environment. :func:`feed` is used to pass the actions to
  the environment for execution.
  """

  @property
  @abstractmethod
  def name(self) -> str:
    """Name of the bandit environment"""

  @abstractmethod
  def reset(self):
    """Reset the bandit environment"""

  @abstractmethod
  def context(self):
    """Current state of the bandit environment"""

  @abstractmethod
  def feed(self, actions):
    """
    Args:
      actions: actions to take

    Return:
      feedback after actions are taken
    """
