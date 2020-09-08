from abc import ABC, abstractmethod


class Learner(ABC):
  """Learner class

  Before a game runs, a learner should be initialized with :func:`reset`.
  """
  def __init__(self, name: str):
    """
    Args:
      name: alias name
    """
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """Learner name"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default learner name
    """

  @abstractmethod
  def reset(self):
    """Learner reset

    Initialization. This function should be called before the start of the game.
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
