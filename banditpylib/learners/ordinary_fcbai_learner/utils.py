from abc import abstractmethod

from typing import Optional, Union, List

from banditpylib.bandits import OrdinaryBandit
from banditpylib.data_pb2 import Arm
from banditpylib.learners import Learner, Goal, IdentifyBestArm


class OrdinaryFCBAILearner(Learner):
  """Base class for bai learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with fixed confidence.

  :param int arm_num: number of arms
  :param float confidence: confidence level. It should be within (0, 1). The
    algorithm should output the best arm with probability at least this value.
  :param str name: alias name
  """
  def __init__(self, arm_num: int, confidence: float, name: Optional[str]):
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

  @property
  def arm_num(self) -> int:
    """Number of arms"""
    return self.__arm_num

  @property
  def confidence(self) -> float:
    """Confidence level of the learner"""
    return self.__confidence

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Index of the best arm identified by the learner"""

  @property
  def goal(self) -> Goal:
    arm = Arm()
    arm.id = self.best_arm
    return IdentifyBestArm(best_arm=arm)
