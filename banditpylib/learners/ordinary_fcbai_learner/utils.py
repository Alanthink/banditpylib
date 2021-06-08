from abc import abstractmethod

from typing import Optional, Union, List

from banditpylib.bandits import OrdinaryBandit
from banditpylib.learners import Learner, Goal, BestArmId


class OrdinaryFCBAILearner(Learner):
  """Base class for bai learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with fixed confidence.
  """
  def __init__(self, arm_num: int, confidence: float, name: Optional[str]):
    """
    Args:
      arm_num: number of arms
      confidence: confidence level. It should be within (0, 1). The algorithm
        should output the best arm with probability at least this value.
      name: alias name
    """
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    self.__arm_num = arm_num
    if confidence <= 0 or confidence >= 1:
      raise ValueError(
          'Confidence level is expected within range (0, 1). Got %.2f.' %
          confidence)
    self.__confidence = confidence

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return OrdinaryBandit

  def arm_num(self) -> int:
    """
    Returns:
      number of arms
    """
    return self.__arm_num

  def confidence(self) -> float:
    """
    Returns:
      confidence level of the learner
    """
    return self.__confidence

  @abstractmethod
  def best_arm(self) -> int:
    """
    Returns:
      index of the best arm identified by the learner
    """

  @property
  def goal(self) -> Goal:
    return BestArmId(best_arm=self.best_arm())
