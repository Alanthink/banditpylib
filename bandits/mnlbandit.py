from abc import abstractmethod
from absl import logging

import numpy as np

from bandits.bandit import BanditEnvironment
from utils import search_best_assortment


class MNLBandit(BanditEnvironment):
  """MNL bandit model

  Products are numbered from 1 by default. 0 is for non-purchase.
  We are assuming abstraction parameter of non-purchase is 1.
  """
  def init(self):
    self.__max_revenue = 0

  @property
  @abstractmethod
  def prod_num(self):
    """Total number of products"""

  @property
  @abstractmethod
  def card_constraint(self):
    pass

  @property
  @abstractmethod
  def type(self):
    pass

  @property
  @abstractmethod
  def context(self):
    pass

  @abstractmethod
  def _best_ans(self, context):
    pass

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
    # remove duplicate products if possible
    assortment = list(set(assortment))

    for prod in assortment:
      if not isinstance(prod, int):
        logging.fatal('Product index should be an integer!')
      if prod < 1 or prod > self.prod_num:
        logging.fatal('Product index should be between 1 and %d!' % self.prod_num)

    if len(assortment) > self.card_constraint:
      logging.fatal('The assortment has products more than %d!' % self.card_constraint)

    best_rev, _, abspar, revenue= self._best_ans(context)

    denominator = sum([abspar[prod] for prod in assortment]) + 1
    prob = [1/denominator] + [abspar[prod]/denominator for prod in assortment]
    observation = np.random.choice(len(prob), 1, p=prob)[0]

    self.__max_revenue += best_rev

    # return (purchase observation, revenue)
    if observation == 0:
      return (0, 0)
    return assortment[observation-1], revenue[assortment[observation-1]]

  def regret(self, rewards):
    revenue = rewards
    del rewards
    return self.__max_revenue - revenue


class OrdinaryMNLBandit(MNLBandit):
  def __init__(self, abspar, revenue, K=np.inf):
    """
    abspar: abstraction parameters of products
    revenue: revenue of products
    K: the cardinality upper bound of every assortment
    """
    logging.info('Ordinary MNL bandit model')
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

    self.__abspar = abspar
    self.__revenue = revenue
    self.__K = K
    self.__product_num = len(abspar)

    # compute the best assortment
    self.__best_rev, self.__best_assort = search_best_assortment([1]+self.__abspar, [0]+self.__revenue, self.__K)
    logging.info('Assortment %s has best revenue: %.3f.' % (self.__best_assort, self.__best_rev))

    self.__type = 'ordinarymnlbandit'

  @property
  def prod_num(self):
    return self.__product_num

  @property
  def card_constraint(self):
    return self.__K

  @property
  def type(self):
    return self.__type

  @property
  def context(self):
    return self.__revenue

  def _best_ans(self, context):
    return (self.__best_rev, self.__best_assort, self.__abspar, self.__revenue)
