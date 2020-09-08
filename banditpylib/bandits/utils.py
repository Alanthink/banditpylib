from abc import ABC, abstractmethod


class Bandit(ABC):
  """Bandit environment

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
    """Bandit name"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """

  @abstractmethod
  def reset(self):
    """Reset the bandit environment

    Initialization. This function should be called before the start of the game.
    """

  @abstractmethod
  def context(self):
    """
    Returns:
      current state of the bandit environment
    """

  @abstractmethod
  def feed(self, actions):
    """
    Args:
      actions: actions for the bandit environment to take

    Returns:
      feedback after executing actions `actions`
    """
