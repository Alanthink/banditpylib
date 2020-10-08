from abc import abstractmethod

import copy
from typing import List, Tuple, Set, Optional

from absl import logging

import numpy as np

from .utils import Bandit


def search(assortments: List[Set[int]],
           product_num: int,
           next_product_id: int,
           assortment: Set[int],
           card_limit: int = np.inf,
           restricted_products: Set[int] = None):
  """Find all assortments satisfying cardinality limit

  Args:
    assortments: all eligible assortments found so far
    product_num: total number of products
    next_product_id: next product to consider
    assortment: current assortment
    card_limit: cardinality limit
    restricted_products: products can only be selected from this restricted set
  """
  if next_product_id == (product_num + 1):
    # ignore empty assortment
    if assortment:
      assortments.append(assortment)
    return
  if len(assortment) < card_limit and (restricted_products is None or
                                       next_product_id in restricted_products):
    search(assortments, product_num, next_product_id + 1,
           assortment.union({next_product_id}), card_limit, restricted_products)
  search(assortments, product_num, next_product_id + 1, assortment, card_limit,
         restricted_products)


class Reward:
  """General reward class"""
  def __init__(self, name: Optional[str]):
    """
    Args:
      name: alias name
    """
    self.__name = self._name() if name is None else name
    self.__preference_params = None
    self.__revenues = None

  @property
  def name(self) -> str:
    """reward name"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default reward name
    """

  @abstractmethod
  def calc(self, assortment: Set[int]) -> float:
    """
    Args:
      assortment: assortment to calculate

    Returns:
      reward of the assortment
    """

  @property
  def preference_params(self) -> np.ndarray:
    """preference parameters (product 0 is included)"""
    if self.__preference_params is None:
      raise Exception('Preference parameters are not set yet!')
    return self.__preference_params

  @property
  def revenues(self) -> np.ndarray:
    """revenues (product 0 is included)"""
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
  def __init__(self, name: str = None):
    """
    Args:
      name: alias name
    """
    super().__init__(name)

  def _name(self) -> str:
    """
    Returns:
      default reward name
    """
    return 'mean_reward'

  def calc(self, assortment: Set[int]) -> float:
    """
    Args:
      assortment: assortment to calculate

    Returns:
      reward of the assortment
    """
    preference_params_sum = (
        sum([self.preference_params[product]
             for product in assortment]) + self.preference_params[0])
    return sum([
        self.preference_params[product] / preference_params_sum *
        self.revenues[product] for product in assortment
    ])


class CvarReward(Reward):
  """CVaR reward"""
  def __init__(self, alpha: float, name: str = None):
    """
    Args:
      alpha: percentile of cvar
      name: alias name
    """
    super().__init__(name)
    if alpha <= 0:
      raise Exception('Alpha %.2f is no greater than 0!' % alpha)
    # alpha is at most 1.0
    if alpha > 1.0:
      logging.error(
          'Alpha %.2f is greater than 1! I am setting it to 1.' % alpha)
    self.__alpha = alpha if alpha <= 1.0 else 1.0

  def _name(self) -> str:
    """
    Returns:
      default reward name
    """
    return 'cvar_reward'

  @property
  def alpha(self) -> float:
    """percentile of cvar"""
    return self.__alpha

  def calc(self, assortment: Set[int]) -> float:
    """
    Args:
      assortment: assortment to calculate

    Returns:
      reward of the assortment
    """
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
                           card_limit: int = np.inf) -> Tuple[float, Set[int]]:
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
    best_assortment = {sorted_revenues[-1][1]}
    next_ind = product_num - 1
    while next_ind > 0 and reward.calc(best_assortment) < revenues[
        sorted_revenues[next_ind][1]]:
      best_assortment.add(sorted_revenues[next_ind][1])
      next_ind -= 1
    if len(best_assortment) <= card_limit:
      return (reward.calc(best_assortment), best_assortment)
    else:
      restricted_products = best_assortment

  assortments: List[Set[int]] = []
  search(assortments=assortments,
         product_num=product_num,
         next_product_id=1,
         assortment=set(),
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
    random_neighbors: int,
    card_limit: int,
    init_assortment: Set[int] = None) -> Tuple[float, Set[int]]:
  """Local search assortment with the maximum reward

  .. warning::
    This method does not guarantee to output the best assortment.

  .. todo::
    Implement this function with `cppyy`.

  Args:
    reward: reward definition
    random_neighbors: number of random neighbors to look up
    card_limit: cardinality constraint
    init_assortment: initial assortment to start

  Returns:
    local best assortment with its reward
  """
  if random_neighbors <= 0:
    raise Exception('Number of neighbors to look up %d is no greater than 0!' \
        % random_neighbors)

  product_num = len(reward.revenues) - 1

  # all available products
  all_products = set(range(1, product_num + 1))
  if init_assortment is None:
    # randomly generate an assortment initially
    best_assortment = set(
        np.random.choice(list(all_products), card_limit, replace=False))
    best_reward = reward.calc(best_assortment)
  else:
    best_assortment = set(init_assortment)
    best_reward = reward.calc(best_assortment)
  remaining_products = all_products - best_assortment

  while True:
    available_operations = []
    if len(remaining_products) > 0:
      available_operations.append('replace')
    if len(best_assortment) > 1:
      available_operations.append('remove')
    if len(remaining_products) > 0 and len(best_assortment) < card_limit:
      available_operations.append('add')

    local_best_assortment = set()
    local_best_reward = 0.0
    for _ in range(random_neighbors):
      # pylint: disable=no-member
      operation = np.random.choice(available_operations)

      if operation == 'replace':
        # replace one product
        product_to_remove = np.random.choice(list(best_assortment))
        product_to_add = np.random.choice(list(remaining_products))
        new_assortment = set(best_assortment)
        new_assortment.remove(product_to_remove)
        new_assortment.add(product_to_add)
        new_reward = reward.calc(new_assortment)
      elif operation == 'remove':
        # remove one product
        product_to_remove = np.random.choice(list(best_assortment))
        new_assortment = set(best_assortment)
        new_assortment.remove(product_to_remove)
        new_reward = reward.calc(new_assortment)
      else:
        # operation = 'add'
        # add one product
        product_to_add = np.random.choice(list(remaining_products))
        new_assortment = set(best_assortment)
        new_assortment.add(product_to_add)
        new_reward = reward.calc(new_assortment)

      if new_reward > local_best_reward:
        local_best_assortment = new_assortment
        local_best_reward = new_reward

    if local_best_reward > best_reward:
      best_assortment = local_best_assortment
      best_reward = local_best_reward
      remaining_products = all_products - best_assortment
    else:
      break

  return (best_reward, best_assortment)


class OrdinaryMNLBandit(Bandit):
  r"""Ordinary MNL bandit

  There are a total of :math:`N` products, where products are numbered from 1 by
  default. During each time step :math:`t`, when an assortment :math:`S_t` which
  is a subset of products is served, the online customer will make a choice
  i.e., whether to buy a product or purchase nothing. The choice is modeled by

  .. math::
    \mathbb{P}(c_t = i) = \frac{v_i}{\sum_{i \in S_t \cup \{0\} } v_i}

  where 0 is reserved for non-purchase and :math:`v_0 = 1`. It is also assumed
  that preference parameters are within the range :math:`[0, 1]`.

  Suppose the rewards are :math:`(r_0, \dots, r_N)`, where :math:`r_0` is always
  0. Let :math:`F(S)` be the cumulative function of the rewards when :math:`S`
  is served. Let :math:`U` be a quasiconvex function denoting the reward the
  learner wants to maximize. The regret is defined as

  .. math::
    T U(F(S^*)) - \sum_{t = 1}^T U(F(S_t))

  where :math:`S^*` is the optimal assortment.
  """
  def __init__(self,
               preference_params: np.ndarray,
               revenues: np.ndarray,
               card_limit: int = np.inf,
               reward: Reward = None,
               zero_best_reward: bool = False,
               name: str = None):
    """
    Args:
      preference_params: preference parameters (product 0 should be included)
      revenue: revenue of products (product 0 should be included)
      card_limit: cardinality constraint of an assortment meaning the total
        number of products provided at a time is no greater than this number
      reward: reward the learner wants to maximize. The default goal is mean of
        rewards
      zero_best_reward: whether to set the reward of the best assortment to 0.
        This is useful when data is too large to compute the best assortment.
        When best reward is set to zero, the regret equals to the minus total
        revenue.
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
    # product 0 is reserved for non-purchase
    self.__product_num = len(self.__preference_params) - 1
    if self.__product_num == 0:
      raise Exception('Number of products %d is 0!' % self.__product_num)
    if card_limit < 1:
      raise Exception('Cardinality limit %d is less than 1!' %
                      card_limit)
    self.__card_limit = min(card_limit, self.__product_num)

    # maximizing the mean of rewards is the default goal
    self.__reward = MeanReward() if reward is None else copy.deepcopy(reward)
    self.__reward.set_preference_params(self.__preference_params)
    self.__reward.set_revenues(self.__revenues)

    if zero_best_reward:
      self.__best_reward, self.__best_assort = 0.0, set()
      logging.warning('Best reward is set to zero. Now the regret equals to the'
                      ' minus total revenue.')
    else:
      # compute the best assortment
      self.__best_reward, self.__best_assort = search_best_assortment(
          reward=self.__reward,
          card_limit=self.__card_limit)
      logging.info('Assortment %s has best reward %.2f.',
                   sorted(list(self.__best_assort)),
                   self.__best_reward)

  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """
    return 'ordinary_mnl_bandit'

  def _take_action(self, assortment: Set[int], times: int) -> \
      Tuple[np.ndarray, List[int]]:
    """Serve one assortment

    Args:
      assortment: assortment to serve
      times: number of serving times

    Returns:
      feedback by serving `assortment`. The first dimension is the
        stochatic rewards, and the second dimension is the choices of the
        customer.
    """
    if not assortment:
      raise Exception('Empty assortment!')
    for product_id in assortment:
      if product_id < 1 or product_id > self.__product_num:
        raise Exception('Product id %d is out of range [1, %d]!' %
                        (product_id, self.__product_num))
    if len(assortment) > self.__card_limit:
      raise Exception('Assortment %s has products more than cardinality'
                      ' constraint %d!' % (sorted(list(assortment)),
                                           self.__card_limit))

    preference_params_sum = sum(
        [self.__preference_params[product_id] for product_id in assortment]) +\
        self.__preference_params[0]
    sorted_assort = sorted(list(assortment))
    sample_prob = [self.__preference_params[0] / preference_params_sum] + \
        [self.__preference_params[product] / preference_params_sum
         for product in sorted_assort]
    sample_results = np.random.choice(len(sample_prob), times, p=sample_prob)
    choices = [
        0 if (sample == 0) else sorted_assort[sample - 1]
        for sample in sample_results
    ]
    # feedback = (stochastic rewards, choices)
    feedback = ([self.__revenues[choice] for choice in choices], choices)

    # update regret
    self.__regret += \
        (self.__best_reward - self.__reward.calc(assortment)) * times
    return feedback

  def feed(self, actions: List[Tuple[Set[int], int]]) -> \
      List[Tuple[np.ndarray, List[int]]]:
    """Serve multiple assortments

    Args:
      actions: for each tuple, the first dimension is the assortment to serve
        and the second dimension is the number of serving times

    Returns:
      feedback by taking `actions`. For each tuple, the first dimension is
        the stochatic rewards, and the second dimension is the choices of the
        customer.
    """
    feedback = []
    for (assortment, times) in actions:
      feedback.append(self._take_action(assortment, times))
    return feedback

  def reset(self):
    """Reset the bandit environment

    .. warning::
      This function should be called before the start of the game.
    """
    self.__regret = 0.0

  def context(self):
    """
    Returns:
      current state of the ordinary mnl bandit
    """
    return None

  def revenues(self) -> np.ndarray:
    """
    Returns:
      revenues of products (product 0 is included, which is always 0.0)
    """
    return self.__revenues

  def product_num(self) -> int:
    """
    Returns:
      number of products (not including product 0)
    """
    return self.__product_num

  def card_limit(self) -> int:
    """
    Returns:
      cardinality limit
    """
    return self.__card_limit

  def regret(self) -> float:
    """
    Returns:
      regret compared with the optimal policy
    """
    return self.__regret
