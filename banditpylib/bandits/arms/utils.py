from abc import ABC, abstractmethod


class Arm(ABC):
  """Abstract class for an arm"""

  @abstractmethod
  def pull(self, pulls=1):
    """
    Args:
      pulls (int, optional): number of pulls

    Return:
      numpy array: stochastic rewards
    """

  @property
  @abstractmethod
  def mean(self):
    """float: mean of arm"""


class EmArm:
  """Class for storing empirical information of an arm"""

  def __init__(self):
    self.reset()

  @property
  def pulls(self):
    """int: total number of pulls"""
    return self.__pulls

  @property
  def rewards(self):
    """float: total rewards obtained so far
    """
    return self.__rewards

  @property
  def em_mean(self):
    """float: empirical mean
    """
    if self.__pulls == 0:
      raise Exception('No empirical mean yet!')
    return self.__rewards / self.__pulls

  @property
  def em_var(self):
    """float: empirical variance
    """
    if self.__pulls == 0:
      raise Exception('No empirical variance yet!')
    return (self.__sq_rewards-self.__rewards**2/self.__pulls)/self.__pulls

  def reset(self):
    """Reset self. Clear historical records"""
    self.__pulls = 0
    self.__rewards = 0
    self.__sq_rewards = 0

  def update(self, *args):
    """Update

    Args:
      args ([float] or [numpy array, int]): the first element is the empirical
        rewards and second element is the number of pulls (if any)
    """
    if len(args) == 1:
      self.__pulls += 1
      self.__rewards += args[0]
      self.__sq_rewards += args[0]**2
    elif len(args) == 2:
      self.__pulls += args[1]
      self.__rewards += sum(args[0])
      self.__sq_rewards += sum(args[0]**2)
    else:
      raise Exception('Update must take 1 or 2 arguments!')
