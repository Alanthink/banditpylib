import math

from typing import Optional

import numpy as np

from .utils import Arm


class PseudoArm(Arm):
  """Pseudo arm

  This class is used to store empirical information of an arm.

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str] = None):
    super().__init__(name)
    self.reset()

  def _name(self) -> str:
    return 'pseudo_arm'

  @property
  def total_pulls(self) -> int:
    """Total number of pulls"""
    return self.__total_pulls

  @property
  def total_rewards(self) -> float:
    """Total rewards obtained so far"""
    return self.__total_rewards

  @property
  def em_mean(self) -> float:
    """Empirical mean of rewards"""
    if self.__total_pulls == 0:
      raise Exception('Number of pulls is 0. No empirical mean.')
    return self.__total_rewards / self.__total_pulls

  @property
  def em_std(self) -> float:
    """Empirical standard variance of rewards"""
    if self.__total_pulls == 0:
      raise Exception('Number of pulls is 0. No empirical standard deviation.')
    return math.sqrt(
        (self.__sum_of_square_reward -
         self.__total_rewards**2 / self.__total_pulls) / self.__total_pulls)

  @property
  def em_var(self) -> float:
    """Empirical variance of rewards"""
    if self.__total_pulls == 0:
      raise Exception('Number of pulls is 0. No empirical variance.')
    return (self.__sum_of_square_reward -
            self.__total_rewards**2 / self.__total_pulls) / self.__total_pulls

  def reset(self):
    """Clear information"""
    self.__total_pulls = 0
    self.__total_rewards = 0
    self.__sum_of_square_reward = 0

  def update(self, rewards: np.ndarray):
    """Update information

    Args:
      rewards: empirical rewards
    """
    self.__total_pulls += len(rewards)
    self.__total_rewards += sum(rewards)
    self.__sum_of_square_reward += sum(rewards**2)
