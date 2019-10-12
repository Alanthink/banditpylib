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

  @abstractmethod
  def type(self):
    pass


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
    self._arms = arms

    self._arm_num = len(arms)
    if self._arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    self._best_arm = self._arms[0]
    for arm in self._arms:
      if arm.mean > self._best_arm.mean:
        self._best_arm = arm

    self._type = 'classicbandit'

  @property
  def arm_num(self):
    """return number of arms"""
    return self._arm_num

  @property
  def type(self):
    return self._type

  def pull(self, action):
    """pull arm"""
    if action not in range(self._arm_num):
      logging.fatal('Wrong arm index!')
    return self._arms[action].pull()

  def regret(self, action_num, rewards):
    """regret compared to the best strategy"""
    return self._best_arm.mean * action_num - rewards


class MNLBandit(BanditEnvironment):
  """MNL bandit model

  Products are numbered from 1 to len(abspar) by default. 0 is for non-purchase.
  """

  def __init__(self, abspar, revenue, K=np.inf):
    """
    abspar: abstraction parameters of products
    revenue: revenue of products
    K: the cardinality upper bound of every assortment

    abspar[0] and revenue[0] are reserved for non-purchase
    """
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

    self._abspar = abspar
    self._revenue = revenue
    self._K = K
    self._product_num = len(abspar)-1

    # compute the best revenue
    self._best_rev, self._best_assort = search_best_assortment(self._abspar, self._revenue, self._K)
    logging.info('Best assortment: %s, with revenue: %.3f' % (self._best_assort, self._best_rev))

    self._type = 'mnlbandit'

  @property
  def prod_num(self):
    return self._product_num

  @property
  def revenue(self):
    return self._revenue

  @property
  def K(self):
    return self._K

  @property
  def type(self):
    return self._type

  def pull(self, action):
    """
    Input:
      action: a list of product indexes
    """
    if not isinstance(action, list):
      logging.fatal('Assortment should be given in a list!')
    if not action:
      logging.fatal('Empty assortment!')

    action = action.copy()

    for prod in action:
      if not isinstance(prod, int):
        logging.fatal('Product index should be an integer!')
      if prod < 1 or prod > self._product_num:
        logging.fatal('Product index should be between 1 and %d' % self._product_num)

    # remove duplicate products if possible
    action = list(set(action))
    if len(action) > self._K:
      logging.fatal('The assortment has products more than K!')

    denominator = sum([self._abspar[prod] for prod in action]) + self._abspar[0]
    prob = [self._abspar[0]/denominator] + [self._abspar[prod]/denominator for prod in action]
    observation = np.random.choice(len(prob), 1, p=prob)[0]
    # return (revenue, purchase observation)
    if observation == 0:
      return (0, 0)
    return self._revenue[action[observation-1]], action[observation-1]

  def regret(self, action_num, rewards):
    return self._best_rev * action_num - rewards
