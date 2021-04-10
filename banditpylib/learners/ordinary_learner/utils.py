from typing import Optional

from banditpylib.bandits import OrdinaryBanditItf
from banditpylib.learners import Learner, Goal, MaxReward


# pylint: disable=W0223
class OrdinaryLearner(Learner):
  """Base class for learners in the ordinary multi-armed bandit

  This type of learners aim to maximize the total rewards.
  """
  def __init__(self, arm_num: int, name: Optional[str]):
    """
    Args:
      arm_num: number of arms
      name: alias name
    """
    super().__init__(name)
    if arm_num <= 1:
      raise Exception('Number of arms %d is less then 2!' % arm_num)
    self.__arm_num = arm_num

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

  @property
  def goal(self) -> Goal:
    """goal of the learner"""
    return MaxReward()
