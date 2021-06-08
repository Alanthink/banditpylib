from abc import abstractmethod

from typing import Optional, Union, List

from banditpylib.bandits import OrdinaryBandit
from banditpylib.learners import Learner, Goal, BestArmId


class OrdinaryFBBAILearner(Learner):
  """Abstract class for best-arm identification learners playing with the
  ordinary multi-armed bandit

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

  @abstractmethod
  def best_arm(self) -> int:
    """
    Returns:
      index of the best arm identified by the learner
    """

  @property
  def goal(self) -> Goal:
    return BestArmId(best_arm=self.best_arm())
