from absl import logging
import numpy as np
from arms import BernoulliArm
from arms import LinearArm
from .utils import Bandit

__all__ = ['OrdinaryLinearBandit']

class OrdinaryLinearBandit(Bandit):
  """Ordinary linear bandit model
  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, arms, theta):
    logging.info('Ordinary linear bandit model')
    if not isinstance(arms, list):
      logging.fatal('Arms should be given in a list!')
    for arm in arms:
      if not isinstance(arm, LinearArm):
        logging.fatal('Not a linear arm!')

    self.__theta = np.array(theta)
    self.__arms = arms

    for idx, arm in enumerate(self.__arms):
      if arm.action.shape !=self.theta.shape:
        logging.fatal('The action and global parameter dimensions are unequal!')

    self.__arm_num = len(arms)
    if self.__arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    self.__best_arm_ind = max([(tup[0], np.dot(tup[1].action,self.theta))
        for tup in enumerate(self.__arms)], key=lambda x:x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_ind]

  @property
  def arm_num(self):
    """return number of arms"""
    return self.__arm_num

  @property
  def arms(self):
    return self.__arms

  @property
  def type(self):
    return 'ordinarybandit'

  @property
  def tot_samples(self):
    return self.__tot_samples

  def init(self):
    self.__tot_samples = 0
    self.__max_rewards = 0

  @property
  def context(self):
    return None

  @property
  def theta(self):
    return self.__theta

  def _take_action(self, action):
    is_list = True
    if not isinstance(action, list):
      is_list = False
      action = [(action, 1)]

    rewards = []
    for tup in action:
      ind = tup[0]
      if ind not in range(self.arm_num):
        logging.fatal('Wrong arm index!')
      rewards.append(self.__arms[ind].pull(self.theta, tup[1]))
      self.__tot_samples += tup[1]
      self.__max_rewards += (self.__best_arm.mean * tup[1])

    if not is_list:
      # rewards[0] is a numpy array with size 1
      return (rewards[0][0],)
    return (rewards,)

  def _update_context(self):
    pass

  def regret(self, rewards):
    return self.__max_rewards - rewards

  def best_arm_regret(self, ind):
    return 1 - (self.__best_arm_ind == ind)
