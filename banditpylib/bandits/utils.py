from abc import ABC, abstractmethod

from typing import Any, Tuple

import numpy as np

from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.learners import Goal


class ContextGenerator(ABC):
  """Context generator

  This class is used to generate the context of bandit.
  """
  def __init__(self, arm_num: int, dimension: int):
    """
    Args:
      arm_num: number of actions
      dimension: dimension of the context
    """
    self.__arm_num = arm_num
    self.__dimension = dimension

  @property
  def name(self) -> str:
    """Context generator name"""
    return self._name()

  @property
  def dimension(self) -> int:
    """dimension of the context"""
    return self.__dimension

  @property
  def arm_num(self) -> int:
    """number of actions"""
    return self.__arm_num

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default context generator name
    """

  @abstractmethod
  def reset(self):
    """Reset the context generator"""

  @abstractmethod
  def context(self) -> Tuple[np.ndarray, np.ndarray]:
    """Returns:
      the context and the rewards corresponding to different actions
    """


class RandomContextGenerator(ContextGenerator):
  """Random context generator

  Fill contexts and rewards information with random numbers in [0, 1].
  """
  def __init__(self, arm_num: int, dimension: int):
    """
    Args:
      arm_num: number of actions
      dimension: dimension of the context
    """
    super().__init__(arm_num, dimension)

  def _name(self) -> str:
    """
    Returns:
      default context generator name
    """
    return 'random_context_generator'

  def reset(self):
    """Reset the context generator"""

  def context(self) -> Tuple[np.ndarray, np.ndarray]:
    """Returns:
      the context and the rewards corresponding to different actions
    """
    return (np.random.random(self.dimension), np.random.random(self.arm_num))


class Bandit(ABC):
  """Bandit environment

  :func:`context` is used to fetch the current state of the environment.
  :func:`feed` is used to pass the actions to the environment for execution.
  """
  @property
  def name(self) -> str:
    """bandit name"""
    return self._name()

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
  def feed(self, actions: Actions) -> Feedback:
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
