from typing import List, Tuple

import numpy as np

from banditpylib.arms import Arm
from .ordinary_bandit_itf import OrdinaryBanditItf


class OrdinaryBandit(OrdinaryBanditItf):
  """Class for ordinary bandit

  Arms are indexed from 0 by default.
  """

  def __init__(self, arms: List[Arm], name=None):
    """
    Args:
      arms: arms in ordinary bandit
      name: alias name
    """
    super().__init__(name)
    if len(arms) < 2:
      raise Exception('The number of arms %d is less than 2!' % len(arms))
    self.__arms = arms
    self.__arm_num = len(arms)
    # find the best arm
    self.__best_arm_id = max(
        [(arm_id, arm.mean) for (arm_id, arm) in enumerate(self.__arms)],
        key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """
    return 'ordinary_bandit'

  def _take_action(self, arm_id: int, pulls=1) -> Tuple[np.ndarray, None]:
    """Pull one arm

    Args:
      arm_id: arm to pull
      pulls: number of times to pull

    Returns:
      feedback by pulling arm `arm_id` where the first dimention is the \
      stochstic rewards
    """
    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))
    self.__total_pulls += pulls
    # empirical rewards when arm `arm_id` is pulled for `pulls` times
    em_rewards = self.__arms[arm_id].pull(pulls)
    if em_rewards is not None:
      self.__regret += (self.__best_arm.mean * pulls - sum(em_rewards))
    return (em_rewards, None)

  def feed(self, actions: List[Tuple[int, int]]) -> \
      List[Tuple[np.ndarray, None]]:
    """Pull multiple arms

    Args:
      actions: for each tuple, the first dimension denotes the arm id and the
        second dimension is the number of times to pull.

    Returns:
      feedback by pulling arms `actions`. For each tuple, the first \
      dimention is the stochstic rewards
    """
    feedback = []
    for (arm_id, pulls) in actions:
      feedback.append(self._take_action(arm_id, pulls))
    return feedback

  def _update_context(self):
    pass

  def reset(self):
    """Reset the bandit environment

    Initialization. This function should be called before the start of the game.
    """
    self.__total_pulls = 0
    self.__regret = 0.0

  def context(self):
    return None

  def arm_num(self) -> int:
    return self.__arm_num

  def total_pulls(self) -> int:
    return self.__total_pulls

  def regret(self) -> float:
    """
    Returns:
      regret compared with the optimal policy
    """
    return self.__regret

  def best_arm_regret(self, arm_id: int) -> int:
    """
    Args:
      arm_id: best arm identified by the learner

    Returns:
      regret compared with the best arm
    """
    return int(self.__best_arm_id != arm_id)
