import copy

import numpy as np

from banditpylib.bandits import Reward
from banditpylib.learners import Learner


# pylint: disable=W0223
class RiskAwareMNLLearner(Learner):
  """Base class for risk-aware learners in the ordinary mnl bandit

  Product 0 is reserved for non-purchase. And it is assumed that the abstraction
  parameter for non-purchase is 1.
  """
  def __init__(self,
               revenues: np.ndarray,
               horizon: int,
               reward: Reward,
               card_limit=np.inf,):
    """
    Args:
      revenues: product revenues
      horizon: total number of time steps
      reward: reward the learner wants to maximize
      card_limit: cardinality constraint
    """
    self.__product_num = len(revenues) - 1
    if self.__product_num < 2:
      raise Exception('Number of products %d is less then 2!' %
                      self.__product_num)
    if revenues[0] != 0:
      raise Exception(
          'The revenue of product 0 i.e., %.2f is not 0!' % revenues[0])
    self.__revenues = revenues
    self.__card_limit = card_limit
    if horizon < self.__product_num:
      raise Exception('Horizon %d is less than number of products %d!' % \
          (horizon, self.__product_num))
    self.__horizon = horizon
    self.__reward = copy.deepcopy(reward)
    self.__reward.set_revenues(self.__revenues)

  def product_num(self):
    return self.__product_num

  def revenues(self):
    return self.__revenues

  def card_limit(self):
    return self.__card_limit

  def horizon(self):
    return self.__horizon

  def reward(self):
    return self.__reward

  def set_horizon(self, horizon: int):
    self._horizon = horizon

  def regret(self, bandit) -> float:
    """
    Return:
      regret compared with the optimal policy
    """
    return bandit.regret()
