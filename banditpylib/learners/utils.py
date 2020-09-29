from abc import ABC, abstractmethod

from typing import Optional

class Learner(ABC):
  """Learner class

  Before a game runs, a learner should be initialized with :func:`reset`.
  """
  def __init__(self, name: Optional[str]):
    """
    Args:
      name: alias name
    """
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """learner name"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default learner name
    """

  @property
  @abstractmethod
  def running_environment(self) -> type:
    """type of environment the learner works with"""

  @abstractmethod
  def reset(self):
    """Learner initialization

    This function should be called before the start of the game.
    """

  @abstractmethod
  def actions(self, context):
    """Actions of the learner for one round

    Args:
      context: context of the bandit environment

    Returns:
      actions to take
    """

  @abstractmethod
  def update(self, feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing actions
        returned by :func:`actions`.
    """
