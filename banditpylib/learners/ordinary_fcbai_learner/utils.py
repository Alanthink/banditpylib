from abc import abstractmethod

from banditpylib.bandits import OrdinaryBanditItf
from banditpylib.learners import Learner


# pylint: disable=W0223
class OrdinaryFCBAILearner(Learner):
  """Base class for bai learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with fixed confidence.
  """
  def __init__(self, arm_num: int, confidence: float, name: str):
    """
    Args:
      arm_num: number of arms
      confidence: confidence level. It should be within (0, 1).
      name: alias name
    """
    super().__init__(name)
    if arm_num <= 1:
      raise Exception('Number of arms %d is less then 2!' % arm_num)
    self.__arm_num = arm_num
    if confidence <= 0 or confidence >= 1:
      raise Exception('Confidence level %.2f is not in range (0, 1)!' % \
                      confidence)
    self.__confidence = confidence

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

  def confidence(self) -> int:
    """
    Returns:
      confidence level of the learner
    """
    return self.__confidence

  def set_confidence(self, confidence: int):
    """
    Args:
      confidence: new confidence level of the learner
    """
    self.__confidence = confidence

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
