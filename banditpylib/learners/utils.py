from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any

import numpy as np

from banditpylib.data_pb2 import Actions, Feedback


def argmax_or_min(values: List[float], find_min: bool = False) -> int:
  """Find index with the highest or smallest value

  Args:
    values: a list of values
    find_min: whether to select smallest value

  Returns:
    index with the highest or smallest value. When there is a tie, randomly
    output one of the indexes.
  """
  extremum = min(values) if find_min else max(values)
  indexes = [index for index, value in enumerate(values) if value == extremum]
  return np.random.choice(indexes)


def argmax_or_min_tuple(values: List[Tuple[float, int]],
                        find_min: bool = False) -> int:
  """Find the second element of the tuple with the highest or smallest value

  Args:
    values: a list of tuples
    find_min: whether to select smallest value

  Returns:
    the second element of the tuple with the highest or smallest value.
    When there is a tie, randomly output one of them.
  """
  extremum = min([value for value, _ in values]) if find_min else max(
      [value for value, _ in values])
  indexes = [index for (value, index) in values if value == extremum]
  return np.random.choice(indexes)


class Goal(ABC):
  """Base class for the goal of a learner"""
  def __init__(self, value: Any):
    """
    Args:
      value: value obtained by the learner
    """
    self.__value = value

  @property
  @abstractmethod
  def name(self) -> str:
    """name of the goal"""

  @property
  def value(self):
    """value obtained by the learner"""
    return self.__value


class BestArmId(Goal):
  """Best arm identification"""
  def __init__(self, best_arm: int):
    """
    Args:
      best_arm: best arm identified by the learner
    """
    super().__init__(value=best_arm)

  @property
  def name(self) -> str:
    """name of the goal"""
    return 'best_arm_id'


class MaxReward(Goal):
  """Reward maximization"""
  def __init__(self):
    """
    Args:
      rewards: rewards obtained by the learner
    """
    super().__init__(value=None)

  @property
  def name(self) -> str:
    """name of the goal"""
    return 'reward_maximization'


class MaxCorrectAnswers(Goal):
  """Maximize correct answers"""
  def __init__(self, answers: List[int]):
    """
    Args:
      answers: answers obtained by the learner
    """
    super().__init__(value=answers)

  @property
  def name(self) -> str:
    """name of the goal"""
    return 'max_correct_answers'


class AllCorrect(Goal):
  """Make all answers correct"""
  def __init__(self, answers: List[int]):
    """
    Args:
      answers: answers obtained by the learner
    """
    super().__init__(value=answers)

  @property
  def name(self) -> str:
    """name of the goal"""
    return 'make_all_correct'


class Learner(ABC):
  """Abstract class for learners

  :func:`actions` returns the actions the learner wants to take. :func:`update`
  is used to pass the feedback of the environment to the learner.
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
    """type of environment the learner is running in"""

  @abstractmethod
  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """

  @abstractmethod
  def actions(self, context) -> Actions:
    """Actions of the learner for one round

    Args:
      context: state of the bandit environment

    Returns:
      actions to take
    """

  @abstractmethod
  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """

  @property
  @abstractmethod
  def goal(self) -> Goal:
    """goal of the learner"""
