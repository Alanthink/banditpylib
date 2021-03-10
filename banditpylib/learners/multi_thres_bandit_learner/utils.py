from typing import Optional

from banditpylib.bandits import MultiThresholdingBandit
from banditpylib.learners import Learner


class MultiThresBanditLearner(Learner):
  """Base class for learners in multi-thresholding bandit"""
  def __init__(self, arm_num: int, budget: int, name: Optional[str]):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      name: alias name
    """
    super().__init__(name)
    if arm_num < 2:
      raise Exception('Number of arms %d is less then 2.' % arm_num)
    self.__arm_num = arm_num
    if budget < arm_num:
      raise Exception('Budget %d is less than number of arms %d.' % \
          (budget, arm_num))
    self.__budget = budget

  @property
  def running_environment(self) -> type:
    """type of environment the learner works with"""
    return MultiThresholdingBandit

  def arm_num(self) -> int:
    """
    Returns:
      number of arms
    """
    return self.__arm_num

  def budget(self) -> int:
    """
    Returns:
      budget of the game
    """
    return self.__budget
