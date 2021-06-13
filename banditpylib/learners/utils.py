from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union

from banditpylib.data_pb2 import Actions, Feedback


class Goal(ABC):
  """Abstract class for the goal of a learner

  :param Any value: value obtained by the learner
  """
  def __init__(self, value: Any):
    self.__value = value

  @property
  @abstractmethod
  def name(self) -> str:
    """Name of the goal"""

  @property
  def value(self):
    """Value obtained by the learner"""
    return self.__value


class BestArmId(Goal):
  """Best arm identification

  :param int best_arm: best arm identified by the learner
  """
  def __init__(self, best_arm: int):
    super().__init__(value=best_arm)

  @property
  def name(self) -> str:
    return 'best_arm_id'


class MaxReward(Goal):
  """Reward maximization"""
  def __init__(self):
    super().__init__(value=None)

  @property
  def name(self) -> str:
    return 'reward_maximization'


class MaxCorrectAnswers(Goal):
  """Maximize correct answers

  This is used by thresholding bandit learners.

  :param List[int] answers: answers obtained by the learner
  """
  def __init__(self, answers: List[int]):
    super().__init__(value=answers)

  @property
  def name(self) -> str:
    return 'max_correct_answers'


class AllCorrect(Goal):
  """Make all answers correct

  This is used by thresholding bandit learners.

  :param List[int] answers: answers obtained by the learner
  """
  def __init__(self, answers: List[int]):
    super().__init__(value=answers)

  @property
  def name(self) -> str:
    return 'make_all_correct'


class Learner(ABC):
  """Abstract class for learners

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """Name of the learner"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default learner name
    """

  @property
  @abstractmethod
  def running_environment(self) -> Union[type, List[type]]:
    """Type of bandit environment the learner plays with"""

  @abstractmethod
  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """

  @abstractmethod
  def actions(self, context) -> Actions:
    """Actions of the learner

    Args:
      context: contextual information about the bandit environment

    Returns:
      actions to take
    """

  @abstractmethod
  def update(self, feedback: Feedback):
    """Update the learner

    Args:
      feedback: feedback returned by the bandit environment after
        :func:`actions` is executed
    """

  @property
  @abstractmethod
  def goal(self) -> Goal:
    """Goal of the learner"""
