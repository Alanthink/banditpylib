from abc import ABC, abstractmethod


class Learner(ABC):
  """Learner class

  Before a game runs, a learner should be initialized with :func:`reset`.
  """
  def __init__(self, name: str):
    """
    Args:
      name: alias name for the learner. This is useful for figure plotting.
    """
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """name of the learner"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """Internal name of the learner"""

  @abstractmethod
  def reset(self):
    """Reset of the learner

    This function should be called before the start of the game
    """

  @abstractmethod
  def actions(self, context=None):
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
      feedback: feedback of the bandit environment
    """
