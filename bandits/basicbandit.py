from absl import logging

from arm import Arm
from bandits.bandit import BanditEnvironment


class Bandit(BanditEnvironment):
  """Classic bandit model

  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, arms):
    logging.info('Classic bandit model')
    if not isinstance(arms, list):
      logging.fatal('Arms should be given in a list!')
    for arm in arms:
      if not isinstance(arm, Arm):
        logging.fatal('Not an arm!')
    self.__arms = arms

    self.__arm_num = len(arms)
    if self.__arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    self.__best_arm = self.__arms[0]
    for arm in self.__arms:
      if arm.mean > self.__best_arm.mean:
        self.__best_arm = arm

    self.__type = 'classicbandit'

  def init(self):
    self.__max_rewards = 0

  def context(self):
    return None

  @property
  def arm_num(self):
    """return number of arms"""
    return self.__arm_num

  @property
  def type(self):
    return self.__type

  def pull(self, context, action):
    """pull arm"""
    arm = action
    del action

    if arm not in range(self.__arm_num):
      logging.fatal('Wrong arm index!')
    self.__max_rewards += self.__best_arm.mean
    return self.__arms[arm].pull()

  def regret(self, rewards):
    """regret compared to the best strategy"""
    return self.__max_rewards - rewards
