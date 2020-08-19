from typing import List, Tuple

from absl import logging

import numpy as np

from .utils import Bandit


def search(assortments, n, i, assortment, card_limit=np.inf):
  """Find all assortments with cardinality limit

  Args:
    assortments: all eligible assortments found so far
    n: total number of products
    i: next product to consider
    assortment: current assortment
    card_limit: cardinality limit
  """
  if i == (n+1):
    if assortment:
      assortments.append(assortment)
    return
  if len(assortment) < card_limit:
    search(assortments, n, i+1, assortment+[i], card_limit)
  search(assortments, n, i+1, assortment, card_limit)


def search_best_assortment(abstraction_params, revenues, card_limit=np.inf):
  """Search best assortment

  Args:
    abstraction_params: abstraction parameters of products
    revenues: revenues of products
    card_limit: cardinality constraint
  """
  assortments = []
  search(assortments, len(abstraction_params) - 1, 1, [], card_limit)
  sorted_assort = sorted([(sum([
      abstraction_params[product] /
      (sum([abstraction_params[product]
            for product in assortment]) + abstraction_params[0]) *
      revenues[product] for product in assortment
  ]), assortment) for assortment in assortments],
                         key=lambda x: x[0])
  return sorted_assort[-1]


class OrdinaryMNLBandit(Bandit):
  """Class for ordinary MNL bandit

  Products are numbered from 1 by default. 0 is reserved for non-purchase.
  It is assumed that the abstraction parameter of non-purchase is 1.
  """

  def __init__(self,
               abstraction_params: np.ndarray,
               revenues: np.ndarray,
               card_limit=np.inf):
    """
    Args:
      abstraction_params: abstraction parameters
      revenue: revenue of products
      card_limit: cardinality constraint of an assortment
    """
    if len(abstraction_params) != len(revenues):
      raise Exception(
          'length of abstraction_params: %d does not equal to length of '
          'revenue %d!' % (len(abstraction_params), len(revenues)))
    for (i, param) in enumerate(abstraction_params):
      if param > 1 or param < 0:
        raise Exception('The %d-th abstraction paramter is '
                        'out of range [0, 1]' % (i+1))
    for (i, revenue) in enumerate(revenues):
      if revenue <= 0:
        raise Exception('The %d-th revenue is no greater than 0!' % (i+1))

    self.__name = 'ordinary_mnl_bandit'
    # add no purchase choice
    self.__abstraction_params = [1] + abstraction_params
    self.__revenues = [0] + revenues
    self.__card_limit = card_limit
    self.__product_num = len(abstraction_params)

    # compute the best assortment
    self.__best_revenue, self.__best_assort = search_best_assortment(
        self.__abstraction_params, self.__revenues, self.__card_limit)
    print(self.__best_revenue)
    logging.info('Assortment %s has best revenue %.2f.' %
                 (self.__best_assort, self.__best_revenue))

  @property
  def name(self):
    return self.__name

  def _take_action(self, assortment, times) -> Tuple[np.ndarray, List[int]]:
    """Try one assortment

    Args:
      assortment: assortment to try
      times: number of times to try

    Return:
      feedback: The first dimension is the stochatic rewards, and the second
      dimension are the choices of the customer.
    """
    if not assortment:
      raise Exception('Empty assortment!')
    # remove duplicate products if possible
    assortment = list(set(assortment))
    for product_id in assortment:
      if product_id < 1 or product_id > self.__product_num:
        raise Exception('Product id %d is out of range [1, %d]!' %
                        (product_id, self.__product_num))
    if len(assortment) > self.__card_limit:
      raise Exception('Assortment %s has products more than cardinality'
                      ' constraint %d!' % (assortment, self.__card_limit))

    abstraction_params_sum = sum(
        [self.__abstraction_params[product_id] for product_id in assortment]) +\
        self.__abstraction_params[0]
    sample_prob = [self.__abstraction_params[0] / abstraction_params_sum] + \
        [self.__abstraction_params[product] / abstraction_params_sum
         for product in assortment]
    sample_results = np.random.choice(len(sample_prob), times, p=sample_prob)
    self.__total_pulls += times
    choices = [0 if (sample == 0) else assortment[sample-1]
               for sample in sample_results]
    # feedback = (stochastic rewards, choices)
    feedback = ([self.__revenues[choice] for choice in choices], choices)
    return feedback

  def feed(self, actions: List[Tuple[List[int], int]]) -> \
      List[Tuple[np.ndarray, List[int]]]:
    """Try multiple assortments

    Args:
      actions: For each tuple, the first dimension is the assortment to try and
      the second dimension is the number of times.

    Return:
      feedback: For each tuple, the first dimension is the stochatic rewards,
      and the second dimension are the choices of the customer.
    """
    feedback = []
    for (assortment, times) in actions:
      feedback.append(self._take_action(assortment, times))
    return feedback

  def reset(self):
    self.__total_pulls = 0

  def context(self):
    return None

  def revenues(self) -> np.ndarray:
    """
    Return:
      revenue of products (including no purchase)
    """
    return self.__revenues

  def product_num(self) -> int:
    """
    Return:
      total number of products
    """
    return self.__product_num

  def card_limit(self) -> int:
    """
    Return:
      cardinality constraint
    """
    return self.__card_limit

  def regret(self, revenue) -> float:
    """
    Args:
      revenue: empirical revenue obtained by learner

    Return:
      regret compared with optimal policy
    """
    return self.__best_revenue * self.__total_pulls - revenue
