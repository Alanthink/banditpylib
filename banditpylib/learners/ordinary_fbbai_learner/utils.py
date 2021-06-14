from abc import abstractmethod

from typing import Optional, Union, List

from banditpylib.bandits import OrdinaryBandit
from banditpylib.data_pb2 import Arm
from banditpylib.learners import Learner, Goal, IdentifyBestArm


class OrdinaryFBBAILearner(Learner):
  """Abstract class for best-arm identification learners playing with the
  ordinary multi-armed bandit

  This kind of learners aim to identify the best arm with fixed budget.

  :param int arm_num: number of arms
  :param int budget: total number of pulls
  :param Optional[str] name: alias name
  """
  def __init__(self, arm_num: int, budget: int, name: Optional[str]):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    self.__arm_num = arm_num
    if budget < arm_num:
      raise ValueError('Budget is expected at least %d. Got %d.' %
                       (self.__arm_num, budget))
    self.__budget = budget

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return OrdinaryBandit

  @property
  def arm_num(self) -> int:
    """Number of arms"""
    return self.__arm_num

  @property
  def budget(self) -> int:
    """Budget of the learner"""
    return self.__budget

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Index of the best arm identified by the learner"""

  @property
  def goal(self) -> Goal:
    arm = Arm()
    arm.id = self.best_arm
    return IdentifyBestArm(best_arm=arm)
