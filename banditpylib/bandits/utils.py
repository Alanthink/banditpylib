from abc import ABC, abstractmethod

from typing import Any, List, Optional, Tuple

from banditpylib.learners import Goal


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
  def context(self) -> Any:
    """
    Returns:
      current state of the bandit environment
    """

  @abstractmethod
  def feed(self, actions: List[Tuple[Any, int]]) -> \
      Optional[List[Tuple[Any, Any]]]:
    """
    Args:
      actions: actions for the bandit environment to execute

    Returns:
      feedback after `actions` are executed
    """

  @abstractmethod
  def regret(self, goal: Goal) -> float:
    """
    Args:
      goal: goal of the learner

    Returns:
      regret of the learner
    """
