from abc import ABC, abstractmethod


class Bandit(ABC):
  """Class for bandit environment

  Before a game is run, one has to call :func:`reset` to initialize the
  environment. During each time step, :func:`context` will return the
  current state of the environment. :func:`feed` is used to pass the actions to
  the environment for execution.
  """
  def __init__(self, name: str):
    """
    Args:
      name: alias name for the bandit environment.
    """
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """name of the learner"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """Internal name of the bandit environment"""

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

    Returns:
      feedback after actions are taken
    """
