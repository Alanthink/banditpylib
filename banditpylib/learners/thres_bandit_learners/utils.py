from typing import Optional

from banditpylib.bandits import ThresholdingBandit
from banditpylib.learners import Learner


class ThresBanditLearner(Learner):
  """Base class for learners in thresholding bandit"""
  def __init__(self, arm_num: int, name: Optional[str]):
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

  @property
  def running_environment(self) -> type:
    """type of environment the learner works with"""
    return ThresholdingBandit

  def arm_num(self) -> int:
    """
    Returns:
      number of arms
    """
    return self.__arm_num
