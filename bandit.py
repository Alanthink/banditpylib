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
  def init(self):
    pass

  @abstractmethod
  def context(self):
    pass

  @abstractmethod
  def pull(self, context, action):
    pass

  @abstractmethod
  def regret(self, rewards):
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


class MNLBandit(BanditEnvironment):
  """MNL bandit model

  Products are numbered from 1 to len(abspar) by default. 0 is for non-purchase.
  """

  def __init__(self, abspar, revenue, K=np.inf):
    """
    abspar: abstraction parameters of products
    revenue: revenue of products
    K: the cardinality upper bound of every assortment
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

    # abspar[0] and revenue[0] are reserved for non-purchase
    # we are assuming abspar[0]=1 and revenue[0]=0
    self.__abspar = [1] + abspar
    self.__revenue = [0] + revenue
    self.__K = K
    self.__product_num = len(abspar)

    # compute the best revenue
    self.__best_rev, self.__best_assort = search_best_assortment(self.__abspar, self.__revenue, self.__K)
    logging.info('Assortment: %s has best revenue: %.3f' % (self.__best_assort, self.__best_rev))

    self.__type = 'mnlbandit'

  def init(self):
    self.__max_revenue = 0

  def context(self):
    return None

  @property
  def prod_num(self):
    return self.__product_num

  @property
  def revenue(self):
    return self.__revenue

  @property
  def K(self):
    return self.__K

  @property
  def type(self):
    return self.__type

  def pull(self, context, action):
    """
    Input:
      action: a list of product indexes
    """
    assortment = action
    del action

    if not isinstance(assortment, list):
      logging.fatal('Assortment should be given in a list!')
    if not assortment:
      logging.fatal('Empty assortment!')

    assortment = assortment.copy()

    for prod in assortment:
      if not isinstance(prod, int):
        logging.fatal('Product index should be an integer!')
      if prod < 1 or prod > self.__product_num:
        logging.fatal('Product index should be between 1 and %d' % self.__product_num)

    # remove duplicate products if possible
    assortment = list(set(assortment))
    if len(assortment) > self.__K:
      logging.fatal('The assortment has products more than %d!' % self.__K)

    denominator = sum([self.__abspar[prod] for prod in assortment]) + self.__abspar[0]
    prob = [self.__abspar[0]/denominator] + [self.__abspar[prod]/denominator for prod in assortment]
    observation = np.random.choice(len(prob), 1, p=prob)[0]

    self.__max_revenue += self.__best_rev

    # return (revenue, purchase observation)
    if observation == 0:
      return (0, 0)
    return self.__revenue[assortment[observation-1]], assortment[observation-1]

  def regret(self, rewards):
    revenue = rewards
    del rewards
    return self.__max_revenue - revenue
