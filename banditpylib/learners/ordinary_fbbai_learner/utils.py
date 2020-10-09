from abc import abstractmethod

from typing import Optional

from banditpylib.bandits import OrdinaryBanditItf
from banditpylib.learners import Learner, Goal, BestArmId


# pylint: disable=W0223
class OrdinaryFBBAILearner(Learner):
  """Base class for bai learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with fixed budget.
  """
  def __init__(self, arm_num: int, budget: int, name: Optional[str]):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      name: alias name
    """
    super().__init__(name)
    if arm_num <= 1:
      raise Exception('Number of arms %d is less then 2!' % arm_num)
    self.__arm_num = arm_num
    if budget < arm_num:
      raise Exception('Budget %d is less than number of arms %d!' % \
          (budget, arm_num))
    self.__budget = budget

  @property
  def running_environment(self) -> type:
    """type of environment the learner works with"""
    return OrdinaryBanditItf

  def arm_num(self) -> int:
    """
    Returns:
      number of arms
    """
    return self.__arm_num

  def budget(self) -> int:
    """
    Returns:
      budget of the learner
    """
    return self.__budget

  def set_budget(self, budget: int):
    """
    Args:
      budget: new budget of the learner
    """
    self.__budget = budget

  @abstractmethod
  def best_arm(self) -> int:
    """
    Returns:
      index of the best arm identified by the learner
    """

  @property
  def goal(self) -> Goal:
    """
    Returns:
      goal of the learner
    """
    return BestArmId(best_arm=self.best_arm())
