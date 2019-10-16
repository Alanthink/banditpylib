from abc import abstractmethod
from absl import logging

from arm import Arm
from bandits.environment import Environment


class Bandit(Environment):

  def init(self):
    self.__max_rewards = 0

  @property
  @abstractmethod
  def arm_num(self):
    pass

  @property
  @abstractmethod
  def type(self):
    pass

  @abstractmethod
  def context(self):
    pass

  @property
  @abstractmethod
  def _oracle_context(self):
    pass

  @abstractmethod
  def _update_context(self):
    pass

  def _take_action(self, action):
    arm = action
    del action

    best_arm, arms = self._oracle_context
    self.__max_rewards += best_arm.mean

    if arm not in range(self.arm_num):
      logging.fatal('Wrong arm index!')

    return arms[arm].pull()

  def regret(self, rewards):
    return self.__max_rewards - rewards


class OrdinaryBandit(Bandit):
  """Ordinary bandit model
  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, arms):
    logging.info('Ordinary bandit model')
    if not isinstance(arms, list):
      logging.fatal('Arms should be given in a list!')
    for arm in arms:
      if not isinstance(arm, Arm):
        logging.fatal('Not an arm!')
    self.__arms = arms

    self.__arm_num = len(arms)
    if self.__arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    sorted_mean = sorted([(arm.mean, arm) for arm in self.__arms], key=lambda x:x[0])
    self.__best_arm = sorted_mean[-1][1]

    self.__type = 'ordinarybandit'

  @property
  def arm_num(self):
    """return number of arms"""
    return self.__arm_num

  @property
  def type(self):
    return self.__type

  def context(self):
    return None

  @property
  def _oracle_context(self):
    return self.__best_arm, self.__arms

  def _update_context(self):
    pass
