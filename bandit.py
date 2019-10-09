"""
Bandit
"""

from abc import ABC, abstractmethod
from absl import logging

import numpy as np

from arm import Arm
from utils import search_best_assortment


class BanditEnvironment(ABC):
  """Abstract bandit environment"""

  @abstractmethod
  def pull(self, action):
    pass

  @abstractmethod
  def regret(self, action_num, rewards):
    pass

  def get_type(self):
    return self.type


class MNLBandit(BanditEnvironment):
  """MNL bandit model

  Products are numbered from 1 to len(abspar) by default. 0 is for non-purchase.
  """

  def __init__(self, abspar, revenue):
    super().__init__()
    logging.info('MNL bandit model')
    if not isinstance(abspar, list) or not isinstance(revenue, list):
      logging.fatal('Parameters should be given in a list!')

    if len(abspar) != len(revenue):
      logging.fatal('Abstract parameter number does not equal to revenue number!')

    for par in abspar:
      if par > 1 or par < 0:
        logging.fatal('Abstraction parameters are assumed between 0 and 1!')

    for rev in revenue:
      if rev < 0:
        logging.fatal('Product revenue should be at least 0!')

    self.abspar = [1] + abspar
    self.revenue = [0] + revenue
    self.product_num = len(abspar)

    # compute the best revenue
    self.best_rev, self.best_assort = search_best_assortment(self.abspar, self.revenue)

    self.type = 'mnlbandit'

  def pull(self, action):
    """
    Input:
      action: a list of product numbers
    """
    if not isinstance(action, list):
      logging.fatal('Assortment should be given in a list!')
    if not action:
      logging.fatal('Empty assortment!')

    action = action.copy()

    for prod in action:
      if not isinstance(prod, int):
        logging.fatal('Product index should be an integer!')
      if prod < 1 or prod > self.product_num:
        logging.fatal('Product index should be between 1 and %d' % self.product_num)

    # remove duplicate products
    action = list(set(action))
    denominator = sum([self.abspar[prod] for prod in action]) + 1
    prob = [1/denominator] + [self.abspar[prod]/denominator for prod in action]
    observation = np.random.choice(len(prob), 1, p=prob)[0]
    # return (revenue, purchase observation)
    if observation == 0:
      return (0, 0)
    return self.revenue[action[observation-1]], action[observation-1]

  def get_prod_num(self):
    return self.product_num

  def get_revenue(self):
    return self.revenue

  def regret(self, action_num, rewards):
    return self.best_rev * action_num - rewards


class Bandit(BanditEnvironment):
  """Classic bandit model

  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, arms):
    super().__init__()
    logging.info('Classic bandit model')
    if not isinstance(arms, list):
      logging.fatal('Arms should be given in a list!')
    for arm in arms:
      if not isinstance(arm, Arm):
        logging.fatal('Not an arm!')
    self.arms = arms

    self.arm_num = len(arms)
    if self.arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    self.best_arm = self.arms[0]
    for arm in self.arms:
      if arm.mean > self.best_arm.mean:
        self.best_arm = arm

    self.type = 'classicbandit'

  def get_arm_num(self):
    """return numbe of arms"""
    return self.arm_num

  def pull(self, action):
    """pull arm"""
    if action not in range(self.arm_num):
      logging.fatal('Wrong arm index!')
    return self.arms[action].pull()

  def regret(self, action_num, rewards):
    """regret compared to the best strategy"""
    return self.best_arm.mean * action_num - rewards
