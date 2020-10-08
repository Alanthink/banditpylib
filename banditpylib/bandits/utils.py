from abc import ABC, abstractmethod

from typing import Optional

class Bandit(ABC):
  """Bandit environment

  :func:`context` is used to fetch the current state of the environment.
  :func:`feed` is used to pass the actions to the environment for execution.
  """
  def __init__(self, name: Optional[str]):
    """
    Args:
      name: alias name for the bandit environment
    """
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """bandit name"""
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

    .. warning::
      This function should be called before the start of the game.
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
      actions: actions for the bandit environment to execute

    Returns:
      feedback after `actions` are executed
    """
