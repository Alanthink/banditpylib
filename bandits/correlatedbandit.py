from importlib import import_module

from absl import logging
import numpy as np
from .utils import Bandit

__all__ = ['CorrelatedBandit']

ARM_PKG = 'arms'


class CorrelatedBandit(Bandit):
  """Correlated bandit model
  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, pars):
    logging.info('Correlated bandit model')
    if pars['arm']['type'] !=  'CorrelatedArm':
      logging.fatal('Not a correlated arm!')
    actions = pars['arm']['means']
    if not isinstance(actions, list):
      logging.fatal('Means should be given in a list!')
    self.__actions = [np.array(action) for action in actions]
    self.__theta = np.array(pars['param'])
    Arm = getattr(import_module(ARM_PKG), pars['arm']['type'])
    arms = [Arm(np.dot(action,self.__theta)) for action in self.__actions]
    self.__arms = arms

    for _, action in enumerate(self.__actions):
      if action.shape != self.__theta.shape:
        logging.fatal('The action and global parameter dimensions are unequal!')

    self.__arm_num = len(arms)
    if self.__arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    self.__best_arm_ind = max([(tup[0], tup[1].mean)
        for tup in enumerate(self.__arms)], key=lambda x:x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_ind]

  @property
  def arm_num(self):
    """return number of arms"""
    return self.__arm_num

  @property
  def type(self):
    return 'correlatedbandit'

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
  def actions(self):
    return self.__actions

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
      rewards.append(self.__arms[ind].pull(tup[1]))
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
