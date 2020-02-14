from absl import logging

import numpy as np

from .utils import Bandit

__all__ = ['OrdinaryMNLBandit', 'search', 'search_best_assortment']


def search(subsets, n, i, path, K=np.inf):
  if i == n:
    if path:
      subsets.append(path)
    return
  if len(path) < K:
    search(subsets, n, i+1, path+[i], K)
  search(subsets, n, i+1, path, K)


def search_best_assortment(abspar, revenue, K=np.inf):
  # non-purchase is assumed to have abstraction par 1
  subsets = []
  search(subsets, len(abspar), 1, [], K)
  sorted_assort = sorted( [ (sum([abspar[prod]/
      (sum([abspar[prod] for prod in subset])+1)*revenue[prod]
      for prod in subset]), subset) for subset in subsets], key=lambda x:x[0] )
  return sorted_assort[-1]


class OrdinaryMNLBandit(Bandit):
  def __init__(self, pars):
    """Ordinary MNL bandit model

    Products are numbered from 1 by default. 0 is for non-purchase.
    It is assumed that the abstraction parameter of non-purchase is 1.

    Input:
      abspar: abstraction parameters of products
      revenue: revenue of products
      K: the cardinality upper bound of every assortment
    """
    logging.info('Ordinary MNL bandit model')
    abspar = pars['abspar']
    revenue = pars['revenue']
    if not isinstance(abspar, list) or not isinstance(revenue, list):
      logging.fatal('Parameters should be given in a list!')
    if len(abspar) != len(revenue):
      logging.fatal(
          'Abstract parameter number does not equal to revenue number!')
    for par in abspar:
      if par > 1 or par < 0:
        logging.fatal('Abstraction parameters are assumed between 0 and 1!')
    for rev in revenue:
      if rev <= 0:
        logging.fatal('Product revenue should be greater than 0!')

    self.__abspar = [1]+abspar
    self.__revenue = [0]+revenue
    if 'K' in pars:
      self.__K = pars['K']
    else:
      self.__K = np.inf
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

  def _update_context(self):
    pass

  def init(self):
    self.__max_revenue = 0

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
        logging.fatal('Product index should be between 1 and %d!' %
            self.prod_num)

    if len(assortment) > self.card_constraint:
      logging.fatal('The assortment has products more than %d!' %
          self.card_constraint)

    self.__max_revenue += self.__best_rev

    denominator = sum([self.__abspar[prod] for prod in assortment]) + self.__abspar[0]
    prob = [self.__abspar[0]/denominator] + \
        [self.__abspar[prod]/denominator for prod in assortment]
    rand = np.random.choice(len(prob), 1, p=prob)[0]
    # feedback = (revenue, purchase observation)
    if rand == 0:
       return (0, self.__revenue[0])
    return (self.__revenue[assortment[rand-1]], assortment[rand-1])

  def regret(self, rewards):
    revenue = rewards
    del rewards
    return self.__max_revenue - revenue
