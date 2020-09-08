import copy

from absl import logging
import numpy as np

from banditpylib.bandits import Reward
from banditpylib.learners import Learner


# pylint: disable=W0223
class OrdinaryMNLLearner(Learner):
  """Base class for learners in the ordinary mnl bandit

  Product 0 is reserved for non-purchase. And it is assumed that the preference
  parameter for non-purchase is 1.
  """
  def __init__(self,
               revenues: np.ndarray,
               horizon: int,
               reward: Reward,
               card_limit: int,
               name: str,
               use_local_search: bool,
               random_neighbors: int):
    """
    Args:
      revenues: product revenues
      horizon: total number of time steps
      reward: reward the learner wants to maximize
      card_limit: cardinality constraint
      name: alias name
      use_local_search: whether to use local search for searching the best
        assortment
      random_neighbors: number of random neighbors to look up if local search is
        used
    """
    super().__init__(name)
    self.__product_num = len(revenues) - 1
    if self.__product_num < 2:
      raise Exception('Number of products %d is less then 2!' %
                      self.__product_num)
    if revenues[0] != 0:
      raise Exception(
          'The revenue of product 0 i.e., %.2f is not 0!' % revenues[0])
    self.__revenues = revenues
    if card_limit < 1:
      raise Exception('Cardinality limit %d is less than 1!' % card_limit)
    self.__card_limit = min(card_limit, self.__product_num)
    if horizon < self.__product_num:
      logging.warning('Horizon %d is less than number of products %d!' % \
          (horizon, self.__product_num))
    self.__horizon = horizon
    self.__reward = copy.deepcopy(reward)
    self.__reward.set_revenues(self.__revenues)
    self.__use_local_search = use_local_search
    if 0 <= random_neighbors < 3:
      raise Exception('Times of local search %d is less than 3!' %
                      random_neighbors)
    self.__random_neighbors = random_neighbors

  def product_num(self) -> int:
    """
    Returns:
      product numbers
    """
    return self.__product_num

  def revenues(self) -> np.ndarray:
    """
    Returns:
      revenues of products (product 0 is included)
    """
    return self.__revenues

  def card_limit(self) -> int:
    """
    Returns:
      cardinality limit
    """
    return self.__card_limit

  def horizon(self) -> int:
    """
    Returns:
      horizon of the learner
    """
    return self.__horizon

  @property
  def reward(self) -> Reward:
    """goal of the learner"""
    return self.__reward

  @property
  def use_local_search(self) -> bool:
    """if local search is enabled"""
    return self.__use_local_search

  @property
  def random_neighbors(self) -> int:
    return self.__random_neighbors

  def set_horizon(self, horizon: int):
    """
    Args:
      horizon: new horizon of the learner
    """
    self.__horizon = horizon

  def regret(self, bandit) -> float:
    """
    Returns:
      regret compared with the optimal policy
    """
    return bandit.regret()
