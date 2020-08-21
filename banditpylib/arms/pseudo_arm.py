import numpy as np


class PseudoArm:
  """Class for pseudo arm

  This class is used to store empirical information of a general arm.
  """
  def __init__(self):
    self.reset()

  def total_pulls(self) -> int:
    """
    Return:
      total number of pulls recorded
    """
    return self.__total_pulls

  def total_rewards(self) -> float:
    """
    Return:
      total rewards obtained so far
    """
    return self.__total_rewards

  def em_mean(self) -> float:
    """
    Return:
      empirical mean
    """
    if self.__total_pulls == 0:
      raise Exception('Number of pulls is 0. No empirical mean!')
    return self.__total_rewards / self.__total_pulls

  def em_var(self) -> float:
    """
    Return:
      empirical variance
    """
    if self.__total_pulls == 0:
      raise Exception('Number of pulls is 0. No empirical variance!')
    return (self.__sum_of_square_reward -
            self.__total_rewards**2 / self.__total_pulls) / self.__total_pulls

  def reset(self):
    """Clear records"""
    self.__total_pulls = 0
    self.__total_rewards = 0
    self.__sum_of_square_reward = 0

  def update(self, rewards: np.ndarray):
    """Update records

    Args:
      rewards: empirical rewards
    """
    self.__total_pulls += len(rewards)
    self.__total_rewards += sum(rewards)
    self.__sum_of_square_reward += sum(rewards**2)
