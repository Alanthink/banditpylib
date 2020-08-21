from typing import List, Tuple

import numpy as np

from banditpylib.arms import Arm
from .ordinary_bandit_itf import OrdinaryBanditItf


class OrdinaryBandit(OrdinaryBanditItf):
  """Class for ordinary bandit

  Arms are indexed from 0 by default.
  """

  def __init__(self, arms: List[Arm]):
    if len(arms) < 2:
      raise Exception('The number of arms %d is less than 2!' % len(arms))
    self.__arms = arms
    self.__arm_num = len(arms)
    # find the best arm
    self.__best_arm_id = max(
        [(arm_id, arm.mean) for (arm_id, arm) in enumerate(self.__arms)],
        key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  @property
  def name(self) -> str:
    return 'ordinary_bandit'

  def _take_action(self, arm_id: int, pulls=1) -> Tuple[np.ndarray, None]:
    """Pull one arm

    Args:
      arm_id: arm to pull
      pulls: number of times to pull

    Return:
      feedback: first dimention is the stochstic rewards
    """
    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))
    self.__total_pulls += pulls
    return (self.__arms[arm_id].pull(pulls), None)

  def feed(self, actions: List[Tuple[int, int]]) -> \
      List[Tuple[np.ndarray, None]]:
    """Pull multiple arms

    Args:
      actions: For each tuple, the first dimension denotes the arm id and the
      second dimension is the number of times to pull.

    Return:
      feedback: For each tuple, the first dimention is the stochstic rewards.
    """
    feedback = []
    for (arm_id, pulls) in actions:
      feedback.append(self._take_action(arm_id, pulls))
    return feedback

  def _update_context(self):
    pass

  def reset(self):
    self.__total_pulls = 0

  def context(self):
    return None

  def arm_num(self) -> int:
    return self.__arm_num

  def total_pulls(self) -> int:
    return self.__total_pulls

  def regret(self, rewards):
    """
    Args:
      rewards: empirical rewards obtained by the learner

    Return:
      regret compared with the optimal policy
    """
    return self.__best_arm.mean * self.__total_pulls - rewards

  def best_arm_regret(self, arm_id: int):
    """
    Args:
      arm_id: best arm identified by the learner

    Return:
      regret compared with the best arm
    """
    return self.__best_arm_id != arm_id
