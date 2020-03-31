from absl import logging

import numpy as np

from .utils import Bandit


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
  sorted_assort = sorted(
      [(sum([abspar[prod]/(sum([abspar[prod] for prod in subset])+1)
             *revenue[prod] for prod in subset]), subset)
       for subset in subsets],
      key=lambda x: x[0])
  return sorted_assort[-1]


class OrdinaryMNLBandit(Bandit):
  """Class for ordinary MNL bandit

  .. inheritance-diagram:: OrdinaryMNLBandit
    :parts: 1

  Products are numbered from 1 by default. 0 is for non-purchase.
  It is assumed that the abstraction parameter of non-purchase is 1.
  """

  def __init__(self, pars):
    """
    Args:
      pars (dict):

        .. code-block:: yaml

            {
              # abstraction parameters of products
              "abspar": [float, ],
              # revenue of products
              "revenue": [float, ],
              # cardinality upper bound for every assortment
              "K": int
            }
    """
    abspar = pars['abspar']
    revenue = pars['revenue']
    if not isinstance(abspar, list) or not isinstance(revenue, list):
      raise Exception('Parameters should be given in a list!')
    if len(abspar) != len(revenue):
      raise Exception(
          'Abstract parameter number does not equal to revenue number!')
    for par in abspar:
      if par > 1 or par < 0:
        raise Exception('Abstraction parameters are assumed between 0 and 1!')
    for rev in revenue:
      if rev <= 0:
        raise Exception('Product revenue should be greater than 0!')

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

  def reset(self):
    self.__max_revenue = 0

  def _take_action(self, action):
    """
    Args:
      action (int or [int, ]): actions to take
    """
    assortment = action
    del action

    if not isinstance(assortment, list):
      raise Exception('Assortment should be given in a list!')
    if not assortment:
      raise Exception('Empty assortment!')

    assortment = assortment.copy()
    # remove duplicate products if possible
    assortment = list(set(assortment))

    for prod in assortment:
      if not isinstance(prod, int):
        raise Exception('Product index should be an integer!')
      if prod < 1 or prod > self.prod_num:
        raise Exception('Product index should be between 1 and %d!' %
                        self.prod_num)

    if len(assortment) > self.card_constraint:
      raise Exception('The assortment has products more than %d!' %
                      self.card_constraint)

    self.__max_revenue += self.__best_rev

    denominator = sum([self.__abspar[prod] for prod in assortment]) + \
                  self.__abspar[0]
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
