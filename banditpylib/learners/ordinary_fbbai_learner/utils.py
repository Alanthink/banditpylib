from abc import abstractmethod

from typing import Optional

from banditpylib.bandits import OrdinaryBandit
from banditpylib.learners import Learner, Goal, BestArmId


class OrdinaryFBBAILearner(Learner):
  """Base class for best-arm identification learners in the ordinary multi-armed
  bandit

  This kind of learners aim to identify the best arm with fixed budget.
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
      raise Exception('Expected number of arms %d is at least 2.' % arm_num)
    self.__arm_num = arm_num
    if budget < arm_num:
      raise Exception('Expected budget %d is at least number of arms.' %
                      budget)
    self.__budget = budget

  @property
  def running_environment(self) -> type:
    """type of environment the learner works with"""
    return OrdinaryBandit

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
    """goal of the learner"""
    return BestArmId(best_arm=self.best_arm())
