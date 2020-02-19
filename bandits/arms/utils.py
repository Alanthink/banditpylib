from abc import ABC, abstractmethod
from absl import logging

__all__ = ['Arm', 'EmArm']


class Arm(ABC):
  """Base class for an arm"""

  @abstractmethod
  def pull(self, pulls=1):
    pass

  @property
  @abstractmethod
  def mean(self):
    pass

# a useful class
class EmArm:
  """Class for storing empirical information of an arm"""

  def __init__(self):
    self.reset()

  @property
  def pulls(self):
    return self.__pulls

  @property
  def rewards(self):
    return self.__rewards

  @property
  def em_mean(self):
    """get empirical mean"""
    if self.__pulls == 0:
      logging.fatal('No empirical mean yet!')
    return self.__rewards / self.__pulls

  @property
  def em_var(self):
    """get empirical variance"""
    if self.__pulls == 0:
      logging.fatal('No empirical variance yet!')
    return (self.__sq_rewards-self.__rewards**2/self.__pulls)/self.__pulls

  def reset(self):
    """clear historical records"""
    self.__pulls = 0
    self.__rewards = 0
    self.__sq_rewards = 0

  def update(self, *args):
    if len(args) == 1:
      self.__pulls += 1
      self.__rewards += args[0]
      self.__sq_rewards += args[0]**2
    elif len(args) == 2:
      self.__pulls += args[1]
      self.__rewards += sum(args[0])
      self.__sq_rewards += sum(args[0]**2)
    else:
      logging.fatal('Update must take 1 or 2 arguments!')
