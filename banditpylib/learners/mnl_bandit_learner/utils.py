import copy

from typing import Optional, List, Union

import numpy as np

from banditpylib.bandits import MNLBandit
from banditpylib.bandits import Reward
from banditpylib.learners import SinglePlayerLearner, Goal, MaximizeTotalRewards


class MNLBanditLearner(SinglePlayerLearner):
  """Abstract class for learners playing with mnl bandit

  Product 0 is reserved for non-purchase. And it is assumed that the preference
  parameter for non-purchase is 1.

  :param np.ndarray revenues: product revenues
  :param Reward reward: reward the learner wants to maximize
  :param int card_limit: cardinality constraint
  :param bool use_local_search: whether to use local search for searching the
    best assortment
  :param int random_neighbors: number of random neighbors to look up if local
    search is enabled
  :param Optional[str] name: alias name
  """
  def __init__(self, revenues: np.ndarray, reward: Reward, card_limit: int,
               use_local_search: bool, random_neighbors: int,
               name: Optional[str]):
    super().__init__(name)
    self.__product_num = len(revenues) - 1
    if self.__product_num < 2:
      raise ValueError('Number of products is expected at least 2. Got %d.' %
                       self.__product_num)
    if revenues[0] != 0:
      raise ValueError('The revenue of product 0 is expected 0. Got %.2f.' %
                       revenues[0])
    self.__revenues = revenues
    if card_limit < 1:
      raise ValueError('Cardinality limit is expected at least 1. Got %d.' %
                       card_limit)
    self.__card_limit = min(card_limit, self.__product_num)
    self.__reward = copy.deepcopy(reward)
    self.__reward.set_revenues(self.__revenues)
    self.__use_local_search = use_local_search
    if 0 <= random_neighbors < 3:
      raise ValueError(
          'Number of neighbors for local search is expected 3. Got %d.' %
          random_neighbors)
    self.__random_neighbors = random_neighbors

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MNLBandit

  @property
  def product_num(self) -> int:
    """Product numbers (not including product 0)"""
    return self.__product_num

  @property
  def revenues(self) -> np.ndarray:
    """Revenues of products (product 0 is included)"""
    return self.__revenues

  @property
  def card_limit(self) -> int:
    """Cardinality limit"""
    return self.__card_limit

  @property
  def reward(self) -> Reward:
    """Reward the learner wants to maximize"""
    return self.__reward

  @property
  def use_local_search(self) -> bool:
    """Whether local search is enabled"""
    return self.__use_local_search

  @property
  def random_neighbors(self) -> int:
    """Number of random neighbors to look up when local search is enabled"""
    return self.__random_neighbors

  @property
  def goal(self) -> Goal:
    return MaximizeTotalRewards()
