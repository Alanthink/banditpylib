from typing import Optional, List, Union

from banditpylib.bandits import OrdinaryBandit, LinearBandit
from banditpylib.learners import Learner, Goal, MaxReward


class OrdinaryLearner(Learner):
  """Abstract class for learners playing with the ordinary multi-armed bandit

  This type of learners aim to maximize the total collected rewards.

  :param int arm_num: number of arms
  :param Optional[str] name: alias name
  """
  def __init__(self, arm_num: int, name: Optional[str]):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d' %
                       arm_num)
    self.__arm_num = arm_num

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return [OrdinaryBandit, LinearBandit]

  def arm_num(self) -> int:
    """
    Returns:
      number of arms
    """
    return self.__arm_num

  @property
  def goal(self) -> Goal:
    return MaxReward()
