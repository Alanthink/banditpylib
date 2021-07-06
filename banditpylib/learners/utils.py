from abc import ABC, abstractmethod
from typing import Optional, List, Union

from banditpylib.data_pb2 import Context, Arm, Actions, Feedback


class Goal(ABC):
  """Abstract class for the goal of a learner"""
  @property
  @abstractmethod
  def name(self) -> str:
    """Name of the goal"""


class IdentifyBestArm(Goal):
  """Best arm identification

  :param Arm best_arm: best arm identified by the learner
  """
  def __init__(self, best_arm: Arm):
    self.__best_arm = best_arm

  @property
  def name(self) -> str:
    return 'best_arm_id'

  @property
  def best_arm(self) -> Arm:
    return self.__best_arm


class MaximizeTotalRewards(Goal):
  """Reward maximization"""
  @property
  def name(self) -> str:
    return 'reward_maximization'


class MaximizeCorrectAnswers(Goal):
  """Maximize correct answers

  This is used by thresholding bandit learners.

  :param List[int] answers: answers obtained by the learner
  """
  def __init__(self, answers: List[int]):
    self.__answers = answers

  @property
  def name(self) -> str:
    return 'max_correct_answers'

  @property
  def answers(self) -> List[int]:
    return self.__answers


class MakeAllAnswersCorrect(Goal):
  """Make all answers correct

  This is used by thresholding bandit learners.

  :param List[int] answers: answers obtained by the learner
  """
  def __init__(self, answers: List[int]):
    self.__answers = answers

  @property
  def name(self) -> str:
    return 'make_all_correct'

  @property
  def answers(self) -> List[int]:
    return self.__answers


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

  @abstractmethod
  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """

  @property
  @abstractmethod
  def running_environment(self) -> Union[type, List[type]]:
    """Type of bandit environment the learner plays with"""

  @property
  @abstractmethod
  def goal(self) -> Goal:
    """Goal of the learner"""


class SinglePlayerLearner(Learner):
  """Abstract class for single player learners

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    super().__init__(name)

  @abstractmethod
  def actions(self, context: Context) -> Actions:
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
