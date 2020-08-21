import copy

import numpy as np

from banditpylib.bandits import search_best_assortment, Reward
from .utils import RiskAwareMNLLearner


class RiskAwareUCB(RiskAwareMNLLearner):
  """Risk-aware UCB policy

  Product 0 is reserved for non-purchase. And it is assumed that the abstraction
  parameter for non-purchase is 1.
  """
  def __init__(self,
               revenues: np.ndarray,
               horizon: int,
               reward: Reward,
               card_limit=np.inf,
               name=None):
    """
    Args:
      revenues: product revenues
      horizon: total number of time steps
      reward: reward the learner wants to maximize
      card_limit: cardinality constraint
      name: alias name for the learner
    """
    # product 0 is reserved for non-purchase
    self.__product_num = len(revenues) - 1
    if self.__product_num < 2:
      raise Exception('Number of products %d is less then 2!' %
                      self.__product_num)
    self.__revenues = revenues
    self.__card_limit = card_limit
    if horizon < self.__product_num:
      raise Exception('Horizon %d is less than number of products %d!' % \
          (horizon, self.__product_num))
    self._horizon = horizon
    self.__reward = copy.deepcopy(reward)
    self.__reward.set_revenues(self.__revenues)
    self.__name = name if name else 'risk_aware_ucb'

  @property
  def name(self):
    return self.__name

  def reset(self):
    # current time step
    self.__time = 1
    # current episode
    self.__episode = 1
    # number of episodes a product is served until the current episode
    # (exclusive)
    self.__serving_times = np.zeros(self.__product_num + 1)
    # number of times the customer chooses a product until the current episode
    # (exclusive)
    self.__customer_choices = np.zeros(self.__product_num + 1)
    self.__last_actions = None
    self.__last_feedback = None

  def UCB(self) -> np.ndarray:
    """
    Return:
      optimistic estimate of abstraction parameters
    """
    # average choices by the customer per episode
    avg_choices = self.__customer_choices / self.__serving_times
    # temperary result
    tmp_result = 48 * np.log(np.sqrt(self.__product_num) * self.__episode +
                             1) / avg_choices
    ucb = avg_choices + np.sqrt(avg_choices * tmp_result) + tmp_result
    ucb[np.isnan(ucb)] = 1
    ucb = np.minimum(ucb, 1)
    return ucb

  def actions(self, context=None):
    if self.__time > self._horizon:
      self.__last_actions = None
      return self.__last_actions
    # check if last observation is not a non-purchase
    if self.__last_feedback and self.__last_feedback[0][1][0] != 0:
      return self.__last_actions
    # When a non-purchase observation happens, a new episode is started and
    # a new assortment to be served is calculated
    self.__reward.set_abstraction_params(self.UCB())
    # calculate assortment with the maximum reward using optimistic abstraction
    # parameters
    _, best_assortment = search_best_assortment(product_num=self.__product_num,
                                                reward=self.__reward,
                                                card_limit=self.__card_limit)
    self.__last_actions = [(best_assortment, 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__customer_choices[feedback[0][1][0]] += 1
    self.__last_feedback = feedback
    self.__time += 1
    if feedback[0][1][0] == 0:
      for product_id in self.__last_actions[0][0]:
        self.__serving_times[product_id] += 1
      self.__episode += 1
