from abc import abstractmethod
from absl import logging

import numpy as np

from bandits.environment import Environment
from utils import search_best_assortment


class MNLBandit(Environment):
  """MNL bandit model

  Products are numbered from 1 by default. 0 is for non-purchase.
  We are assuming abstraction parameter of non-purchase is 1.
  """
  def init(self):
    self.__max_revenue = 0

  @property
  @abstractmethod
  def prod_num(self):
    pass

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

  @property
  @abstractmethod
  def _oracle_context(self):
    pass

  @abstractmethod
  def _update_context(self):
    pass

  def _take_action(self, action):
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

    _, best_rev, abspar, revenue = self._oracle_context
    self.__max_revenue += best_rev

    denominator = sum([abspar[prod] for prod in assortment]) + abspar[0]
    prob = [abspar[0]/denominator] + [abspar[prod]/denominator for prod in assortment]
    rand = np.random.choice(len(prob), 1, p=prob)[0]
    # feedback = (purchase observation, revenue)
    if rand == 0:
       return (0, revenue[0])
    return (assortment[rand-1], revenue[assortment[rand-1]])

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

    self.__abspar = [1]+abspar
    self.__revenue = [0]+revenue
    self.__K = K
    self.__product_num = len(abspar)

    # compute the best assortment
    self.__best_rev, self.__best_assort = search_best_assortment(
        self.__abspar, self.__revenue, self.__K)
    logging.info('Assortment %s has best revenue %.3f.' %
        (self.__best_assort, self.__best_rev))

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

  @property
  def _oracle_context(self):
    return (self.__best_assort, self.__best_rev, self.__abspar, self.__revenue)

  def _update_context(self):
    pass
