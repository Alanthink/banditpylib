from abc import abstractmethod

from typing import List, Tuple, Set, Optional

from absl import logging

import numpy as np


def search(
    assortments: List[Set[int]],
    product_num: int,
    next_product_id: int,
    assortment: Set[int],
    card_limit: int = np.inf,  # type: ignore
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
           assortment.union({next_product_id}), card_limit,
           restricted_products)
  search(assortments, product_num, next_product_id + 1, assortment, card_limit,
         restricted_products)


class Reward:
  """General reward class

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    self.__name = self._name() if name is None else name
    self.__preference_params: Optional[np.ndarray] = None
    self.__revenues: Optional[np.ndarray] = None

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
  """Mean reward

  :param str name: alias name
  """
  def __init__(self, name: str = None):
    super().__init__(name)

  def _name(self) -> str:
    return 'mean_reward'

  def calc(self, assortment: Set[int]) -> float:
    preference_params_sum = (
        sum([self.preference_params[product]
             for product in assortment]) + self.preference_params[0])
    return sum([
        self.preference_params[product] / preference_params_sum *
        self.revenues[product] for product in assortment
    ])


class CvarReward(Reward):
  """CVaR reward

  :param float alpha: percentile of cvar
  :param str name: alias name
  """
  def __init__(self, alpha: float, name: str = None):
    super().__init__(name)
    if alpha <= 0:
      raise Exception('Alpha %.2f is no greater than 0!' % alpha)
    # alpha is at most 1.0
    if alpha > 1.0:
      logging.error('Alpha %.2f is greater than 1! I am setting it to 1.' %
                    alpha)
    self.__alpha = alpha if alpha <= 1.0 else 1.0

  def _name(self) -> str:
    return 'cvar_reward'

  @property
  def alpha(self) -> float:
    """Percentile of cvar"""
    return self.__alpha

  def calc(self, assortment: Set[int]) -> float:
    preference_params_sum = sum(
        [self.preference_params[product]
         for product in assortment]) + self.preference_params[0]
    # Sort according to revenue of product
    revenue_prob = sorted([
        (self.revenues[0], self.preference_params[0] / preference_params_sum)
    ] + [(self.revenues[product],
          self.preference_params[product] / preference_params_sum)
         for product in assortment],
                          key=lambda x: x[0])
    # The minimum revenue should be 0, which is the revenue of non-purchase
    if revenue_prob[0][0] != 0:
      raise Exception('CVaR calculation error!')
    # Find the index of revenue_prob such that the accumulate probability is at
    # least alpha
    accumulate_prob = revenue_prob[0][1]
    next_ind = 1
    while next_ind < len(revenue_prob) and (accumulate_prob < self.__alpha
                                            or revenue_prob[next_ind][0]
                                            == revenue_prob[next_ind - 1][0]):
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
                           card_limit: float = np.inf
                           ) -> Tuple[float, Set[int]]:
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
  search(
      assortments=assortments,
      product_num=product_num,
      next_product_id=1,
      assortment=set(),
      card_limit=card_limit,  # type: ignore
      restricted_products=restricted_products)
  # Sort assortments according to reward value
  sorted_assort = sorted([(reward.calc(assortment), assortment)
                          for assortment in assortments],
                         key=lambda x: x[0])
  # Randomly select one assortment with the maximum reward
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

  # All available products
  all_products = set(range(1, product_num + 1))
  if init_assortment is None:
    # Randomly generate an assortment initially
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
        # Replace one product
        product_to_remove = np.random.choice(list(best_assortment))
        product_to_add = np.random.choice(list(remaining_products))
        new_assortment = set(best_assortment)
        new_assortment.remove(product_to_remove)
        new_assortment.add(product_to_add)
        new_reward = reward.calc(new_assortment)
      elif operation == 'remove':
        # Remove one product
        product_to_remove = np.random.choice(list(best_assortment))
        new_assortment = set(best_assortment)
        new_assortment.remove(product_to_remove)
        new_reward = reward.calc(new_assortment)
      else:
        # operation = 'add'
        # Add one product
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
