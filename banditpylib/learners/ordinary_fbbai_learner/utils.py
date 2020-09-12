from abc import abstractmethod

from banditpylib.bandits import OrdinaryBanditItf
from banditpylib.learners import Learner


# pylint: disable=W0223
class OrdinaryFBBAILearner(Learner):
  """Base class for bai learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with fixed budget.
  """
  def __init__(self, arm_num: int, budget: int, name: str):
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
    """
    Returns:
      environment class the learner works with
    """
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

  def regret(self, bandit) -> int:
    """
    Returns:
      best arm regret. 0 when the leaner returns the best arm and 1 \
      otherwise
    """
    return bandit.best_arm_regret(self.best_arm())
