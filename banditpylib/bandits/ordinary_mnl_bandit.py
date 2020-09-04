from abc import abstractmethod

import copy
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
  def __init__(self):
    self.__preference_params = None
    self.__revenues = None

  @abstractmethod
  def calc(self, assortment: List[int]) -> float:
    """
    Args:
      assortment: input assortment to calculate

    Returns:
      reward of the assortment
    """

  @property
  def preference_params(self) -> np.ndarray:
    if self.__preference_params is None:
      raise Exception('Preference parameters are not set yet!')
    return self.__preference_params

  @property
  def revenues(self) -> np.ndarray:
    if self.__revenues is None:
      raise Exception('Revenues of products are not set yet!')
    return self.__revenues

  def set_preference_params(self, preference_params: np.ndarray):
    """
    Args:
      preference_params: preference parameters of products
    """
    self.__preference_params = preference_params

  def set_revenues(self, revenues: np.ndarray):
    """
    Args:
      revenues: revenues of products
    """
    self.__revenues = revenues


class MeanReward(Reward):
  """Mean reward"""
  def __init__(self):
    super().__init__()

  def calc(self, assortment: List[int]) -> float:
    preference_params_sum = (
        sum([self.preference_params[product]
             for product in assortment]) + self.preference_params[0])
    return sum([
        self.preference_params[product] / preference_params_sum *
        self.revenues[product] for product in assortment
    ])


class CvarReward(Reward):
  """CVaR reward"""
  def __init__(self, alpha: float):
    """
    Args:
      alpha: percentile of cvar
    """
    super().__init__()
    if alpha <= 0:
      raise Exception('Alpha %.2f is no greater than 0!' % alpha)
    # alpha is at most 1.0
    if alpha > 1.0:
      logging.error(
          'Alpha %.2f is greater than 1! I am setting it to 1.' % alpha)
    self.__alpha = alpha if alpha <= 1.0 else 1.0

  @property
  def alpha(self) -> float:
    return self.__alpha

  def calc(self, assortment: List[int]) -> float:
    preference_params_sum = sum(
        [self.preference_params[product]
         for product in assortment]) + self.preference_params[0]
    # sort according to revenue of product
    revenue_prob = sorted(
        [(self.revenues[0], self.preference_params[0] / preference_params_sum)]+
        [(self.revenues[product],
          self.preference_params[product] / preference_params_sum)
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
    cvar_alpha = sum(
        [revenue_prob[ind][0] * revenue_prob[ind][1] \
        for ind in range(next_ind)])
    cvar_alpha -= revenue_prob[next_ind - 1][0] * (accumulate_prob -
                                                   self.__alpha)
    cvar_alpha /= self.__alpha
    return cvar_alpha


def search_best_assortment(reward: Reward,
                           card_limit=np.inf) -> Tuple[float, List[int]]:
  """Search assortment with the maximum reward

  Args:
    reward: reward definition
    card_limit: cardinality constraint

  Returns:
    assortment with the maximum reward
  """
  product_num = len(reward.revenues) - 1
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


def local_search_best_assortment(
    reward: Reward,
    search_times: int,
    card_limit: int,
    init_assortment=None) -> Tuple[float, List[int]]:
  """Local search assortment with the maximum reward

  .. warning::
    This method does not guarantee to output the best assortment.

  .. todo::
    Implement this function with `cppyy`.

  Args:
    reward: reward definition
    search_times: number of local search times
    card_limit: cardinality constraint
    init_assortment: initial assortment to start

  Returns:
    local best assortment with its reward
  """
  product_num = len(reward.revenues) - 1

  # all available products
  all_products = set(range(1, product_num + 1))
  # randomly generate an assortment initially if init_assortment is not set
  best_assortment = set(init_assortment) if init_assortment else set(
      np.random.choice(
          list(all_products),
          np.random.randint(1, min(card_limit, product_num) + 1),
          replace=False))
  best_reward = reward.calc(best_assortment)
  remaining_products = all_products - best_assortment

  times_of_local_search = 0
  while times_of_local_search < search_times:
    available_operations = []
    if len(remaining_products) > 0:
      available_operations.append('replace')
    if len(best_assortment) > 1:
      available_operations.append('remove')
    if len(remaining_products) > 0 and len(best_assortment) < card_limit:
      available_operations.append('add')
    # pylint: disable=E1101
    operation = np.random.choice(available_operations)

    new_assortment = set(best_assortment)
    if operation == 'replace':
      # replace one product
      product_to_remove = np.random.choice(list(best_assortment))
      product_to_add = np.random.choice(list(remaining_products))
      new_assortment.remove(product_to_remove)
      new_assortment.add(product_to_add)
      new_reward = reward.calc(new_assortment)
      if new_reward > best_reward:
        best_assortment = new_assortment
        best_reward = new_reward
        remaining_products.remove(product_to_add)
    elif operation == 'remove':
      # remove one product
      product_to_remove = np.random.choice(list(best_assortment))
      new_assortment.remove(product_to_remove)
      new_reward = reward.calc(new_assortment)
      if new_reward > best_reward:
        best_assortment = new_assortment
        best_reward = new_reward
        remaining_products.add(product_to_remove)
    else:
      # operation == 'add'
      product_to_add = np.random.choice(list(remaining_products))
      new_assortment.add(product_to_add)
      new_reward = reward.calc(new_assortment)
      if new_reward > best_reward:
        best_assortment = new_assortment
        best_reward = new_reward
        remaining_products.remove(product_to_add)

    times_of_local_search += 1

  return (best_reward, sorted(list(best_assortment)))


class OrdinaryMNLBandit(Bandit):
  """Class for ordinary MNL bandit

  Products are numbered from 1 by default. 0 is reserved for non-purchase. The
  preference parameters are assumed to be between 0 and 1. And it is assumed
  that the preference parameter for non-purchase is 1.
  """
  def __init__(self,
               preference_params: np.ndarray,
               revenues: np.ndarray,
               card_limit=np.inf,
               reward=None,
               zero_best_reward=False,
               name=None):
    """
    Args:
      preference_params: preference parameters
      revenue: revenue of products
      card_limit: cardinality constraint of an assortment
      reward: reward the learner wants to maximize
      zero_best_reward: whether to set the reward of the best assortment to 0. \
      This is useful when data is too large to compute the best assortment.
      name: alias name
    """
    super().__init__(name)
    if len(preference_params) != len(revenues):
      raise Exception(
          'Number of preference parameters %d does not equal to number of '
          'revenues %d!' % (len(preference_params), len(revenues)))
    for (i, param) in enumerate(preference_params):
      if param > 1 or param < 0:
        raise Exception('The %d-th preference parameter is '
                        'out of range [0, 1]' % i)
    if preference_params[0] != 1:
      raise Exception(
          'The preference parameter of product 0 i.e., %.2f is not 1!' %
          preference_params[0])
    for (i, revenue) in enumerate(revenues):
      if i > 0 and revenue <= 0:
        raise Exception('The %d-th revenue is no greater than 0!' % i)
    if revenues[0] != 0:
      raise Exception('The revenue of product 0 i.e., %.2f is not 0!' %
                      revenues[0])

    self.__preference_params = preference_params
    self.__revenues = revenues
    self.__card_limit = card_limit
    # product 0 is reserved for non-purchase
    self.__product_num = len(self.__preference_params) - 1
    if self.__product_num < 1:
      raise Exception('Number of products %d is less than 1!' %
                      self.__product_num)

    # maximizing MeanReward is the default goal
    self.__reward = MeanReward() if reward is None else copy.deepcopy(reward)
    self.__reward.set_preference_params(self.__preference_params)
    self.__reward.set_revenues(self.__revenues)

    if zero_best_reward:
      self.__best_reward, self.__best_assort = 0.0, []
      logging.warning('Best reward is set to zero. Now the regret equals to the'
                      ' minus total revenue.')
    else:
      # compute the best assortment
      self.__best_reward, self.__best_assort = search_best_assortment(
          reward=self.__reward,
          card_limit=self.__card_limit)
      logging.info('Assortment %s has best reward %.2f.', self.__best_assort,
                   self.__best_reward)

  def _name(self):
    return 'ordinary_mnl_bandit'

  def _take_action(self, assortment: List[int], times: int) -> \
      Tuple[np.ndarray, List[int]]:
    """Serve one assortment

    Args:
      assortment: assortment to try
      times: number of times to try

    Returns:
      feedback where the first dimension is the stochatic rewards, and the \
      second dimension are the choices of the customer.
    """
    if not assortment:
      raise Exception('Empty assortment!')
    if len(list(set(assortment))) != len(assortment):
      logging.error('Assortment %s contains duplicate products!' % assortment)
      # remove duplicate products
      assortment = sorted(list(set(assortment)))
    for product_id in assortment:
      if product_id < 1 or product_id > self.__product_num:
        raise Exception('Product id %d is out of range [1, %d]!' %
                        (product_id, self.__product_num))
    if len(assortment) > self.__card_limit:
      raise Exception('Assortment %s has products more than cardinality'
                      ' constraint %d!' % (assortment, self.__card_limit))

    preference_params_sum = sum(
        [self.__preference_params[product_id] for product_id in assortment]) +\
        self.__preference_params[0]
    sample_prob = [self.__preference_params[0] / preference_params_sum] + \
        [self.__preference_params[product] / preference_params_sum
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
      actions: for each tuple, the first dimension is the assortment to try \
      and the second dimension is the number of times.

    Returns:
      feedback where for each tuple, the first dimension is the stochatic \
      rewards, and the second dimension are the choices of the customer
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
    Returns:
      regret compared with the optimal policy
    """
    return self.__regret
