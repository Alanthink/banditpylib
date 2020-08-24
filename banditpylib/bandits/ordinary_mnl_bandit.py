from abc import abstractmethod

from typing import List, Tuple

from absl import logging

import numpy as np

from .utils import Bandit


def search(assortments: List[List[int]],
           product_num: int,
           next_product_id: int,
           assortment: List[int],
           card_limit=np.inf,
           restricted_products=None):
  """Find all assortments with cardinality limit

  Args:
    assortments: all eligible assortments found so far
    product_num: total number of products
    next_product_id: next product to consider
    assortment: current assortment
    card_limit: cardinality limit
    restricted_products: products can only be selected from this restricted set
  """
  if next_product_id == (product_num + 1):
    if assortment:
      assortments.append(assortment)
    return
  if len(assortment) < card_limit and (not restricted_products or
                                       next_product_id in restricted_products):
    search(assortments, product_num, next_product_id + 1,
           assortment + [next_product_id], card_limit, restricted_products)
  search(assortments, product_num, next_product_id + 1, assortment, card_limit,
         restricted_products)


class Reward:
  """Reward class"""
  @abstractmethod
  def calc(self, assortment: List[int]) -> float:
    """
    Args:
      assortment: input assortment to calculate

    Return:
      reward of the assortment
    """

  @property
  def abstraction_params(self) -> np.ndarray:
    if hasattr(self, '__abstraction_params'):
      raise Exception('Abstraction parameters are not set yet!')
    return self.__abstraction_params

  @property
  def revenues(self) -> np.ndarray:
    if hasattr(self, '__revenues'):
      raise Exception('Revenues of products are not set yet!')
    return self.__revenues

  def set_abstraction_params(self, abstraction_params: np.ndarray):
    """
    Args:
      abstraction_params: abstraction parameters of products
    """
    self.__abstraction_params = abstraction_params

  def set_revenues(self, revenues: np.ndarray):
    """
    Args:
      revenues: revenues of products
    """
    self.__revenues = revenues


class MeanReward(Reward):
  """Mean reward"""
  def calc(self, assortment: List[int]) -> float:
    abstraction_params_sum = (
        sum([self.abstraction_params[product]
             for product in assortment]) + self.abstraction_params[0])
    return sum([
        self.abstraction_params[product] / abstraction_params_sum *
        self.revenues[product] for product in assortment
    ])


class CvarReward(Reward):
  """CVaR reward"""
  def __init__(self, alpha: float):
    """
    Args:
      alpha: percentile of cvar
    """
    if alpha <= 0:
      raise Exception('Alpha %.2f is no greater than 0!' % alpha)
    # alpha is at most 1.0
    self.__alpha = alpha if alpha <= 1.0 else 1.0

  def calc(self, assortment: List[int]) -> float:
    abstraction_params_sum = sum(
        [self.abstraction_params[product]
         for product in assortment]) + self.abstraction_params[0]
    # sort according to revenue of product
    revenue_prob = sorted([
        (self.revenues[0], self.abstraction_params[0] / abstraction_params_sum)
    ] + [(self.revenues[product], self.abstraction_params[product])
         for product in assortment],
                          key=lambda x: x[0])
    # the minimum revenue should be 0, which is the revenue of non-purchase
    if revenue_prob[0][0] != 0:
      raise Exception('CVaR calculation error!')
    # find the index of revenue_prob such that the accumulate probability is at
    # least alpha
    accumulate_prob = revenue_prob[0][1]
    next_ind = 1
    while next_ind < len(revenue_prob) and (
        accumulate_prob < self.__alpha
        or revenue_prob[next_ind][0] == revenue_prob[next_ind - 1][0]):
      accumulate_prob += revenue_prob[next_ind][1]
      next_ind += 1
    # calculate cvar_alpha
    cvar_alpha = sum([
        revenue_prob[ind][0] * revenue_prob[ind][1] for ind in range(next_ind)
    ])
    cvar_alpha -= revenue_prob[next_ind - 1][0] * (accumulate_prob -
                                                   self.__alpha)
    cvar_alpha /= self.__alpha
    return cvar_alpha


def search_best_assortment(product_num: int,
                           reward: Reward,
                           card_limit=np.inf):
  """Search assortment with the maximum reward

  Args:
    product_num: product number
    reward: reward definition
    card_limit: cardinality constraint

  Return:
    (reward, assortment): assortment with the maximum reward
  """
  restricted_products = None

  if isinstance(reward, MeanReward):
    # a fast method to find the best assortment when the reward is MeanReward
    revenues = reward.revenues
    sorted_revenues = sorted(list(zip(revenues, range(product_num + 1))),
                             key=lambda x: x[0])
    best_assortment = [sorted_revenues[-1][1]]
    next_ind = product_num - 1
    while next_ind > 0 and reward.calc(best_assortment) < revenues[
        sorted_revenues[next_ind][1]]:
      best_assortment.append(sorted_revenues[next_ind][1])
      next_ind -= 1
    if len(best_assortment) <= card_limit:
      return (reward.calc(best_assortment), sorted(best_assortment))
    else:
      restricted_products = best_assortment

  assortments = []
  search(assortments=assortments,
         product_num=product_num,
         next_product_id=1,
         assortment=[],
         card_limit=card_limit,
         restricted_products=restricted_products)
  # sort assortments according to reward value
  sorted_assort = sorted([(reward.calc(assortment), assortment)
                          for assortment in assortments],
                         key=lambda x: x[0])
  # randomly select one assortment with the maximum reward
  ind = len(sorted_assort) - 1
  while (ind > 0 and sorted_assort[ind - 1][0] == sorted_assort[ind][0]):
    ind -= 1
  return sorted_assort[np.random.randint(ind, len(sorted_assort))]


class OrdinaryMNLBandit(Bandit):
  """Class for ordinary MNL bandit

  Products are numbered from 1 by default. 0 is reserved for non-purchase. The
  abstraction parameters are assumed to be between 0 and 1. And it is assumed
  that the abstraction parameter for non-purchase is 1.
  """
  def __init__(self,
               abstraction_params: np.ndarray,
               revenues: np.ndarray,
               card_limit=np.inf,
               reward=None):
    """
    Args:
      abstraction_params: abstraction parameters
      revenue: revenue of products
      card_limit: cardinality constraint of an assortment
      reward: reward the learner wants to maximize
    """
    if len(abstraction_params) != len(revenues):
      raise Exception(
          'Number of abstraction parameters %d does not equal to number of '
          'revenues %d!' % (len(abstraction_params), len(revenues)))
    for (i, param) in enumerate(abstraction_params):
      if param > 1 or param < 0:
        raise Exception('The %d-th abstraction parameter is '
                        'out of range [0, 1]' % i)
    if abstraction_params[0] != 1:
      raise Exception(
          'The abstraction parameter of product 0 i.e., %.2f is not 1!' %
          abstraction_params[0])
    for (i, revenue) in enumerate(revenues):
      if i > 0 and revenue <= 0:
        raise Exception('The %d-th revenue is no greater than 0!' % i)
    if revenues[0] != 0:
      raise Exception('The revenue of product 0 i.e., %.2f is not 0!' %
                      revenues[0])

    self.__name = 'ordinary_mnl_bandit'
    self.__abstraction_params = abstraction_params
    self.__revenues = revenues
    self.__card_limit = card_limit
    # product 0 is reserved for non-purchase
    self.__product_num = len(self.__abstraction_params) - 1
    if self.__product_num < 1:
      raise Exception('Number of products %d is less than 1!' %
                      self.__product_num)

    # MeanReward is the default goal
    self.__reward = MeanReward() if not reward else reward
    self.__reward.set_abstraction_params(self.__abstraction_params)
    self.__reward.set_revenues(self.__revenues)

    # compute the best assortment
    self.__best_reward, self.__best_assort = search_best_assortment(
        product_num=self.__product_num,
        reward=self.__reward,
        card_limit=self.__card_limit)
    logging.info('Assortment %s has best reward %.2f.', self.__best_assort,
                 self.__best_reward)

  @property
  def name(self):
    return self.__name

  def _take_action(self, assortment, times) -> Tuple[np.ndarray, List[int]]:
    """Serve one assortment

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
    choices = [
        0 if (sample == 0) else assortment[sample - 1]
        for sample in sample_results
    ]
    # feedback = (stochastic rewards, choices)
    feedback = ([self.__revenues[choice] for choice in choices], choices)

    # update regret
    self.__regret += \
        (self.__best_reward - self.__reward.calc(assortment)) * times
    return feedback

  def feed(self, actions: List[Tuple[List[int], int]]) -> \
      List[Tuple[np.ndarray, List[int]]]:
    """Serve multiple assortments

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
    self.__regret = 0.0

  def context(self):
    return None

  def revenues(self) -> np.ndarray:
    return self.__revenues

  def product_num(self) -> int:
    return self.__product_num

  def card_limit(self) -> int:
    return self.__card_limit

  def regret(self) -> float:
    """
    Return:
      regret compared with the optimal policy
    """
    return self.__regret
