from typing import List, Tuple

from absl import logging

import numpy as np

from banditpylib.bandits import search_best_assortment, Reward, \
    local_search_best_assortment
from .utils import OrdinaryMNLLearner


class UCB(OrdinaryMNLLearner):
  """UCB policy :cite:`DBLP:journals/ior/AgrawalAGZ19`"""
  def __init__(self,
               revenues: np.ndarray,
               horizon: int,
               reward: Reward,
               card_limit=np.inf,
               name=None,
               use_local_search=False,
               local_search_times=10):
    """
    Args:
      use_local_search: whether to use local search for searching the best \
      assortment
      local_search_times: number of local searches if local search is used
    """
    super().__init__(revenues, horizon, reward, card_limit, name)
    self.__use_local_search = use_local_search
    if local_search_times < 10:
      logging.fatal('Times of local search %d is less than 10!' %
                    local_search_times)
    self.__local_search_times = local_search_times

  def _name(self):
    return 'risk_aware_ucb'

  def reset(self):
    # current time step
    self.__time = 1
    # current episode
    self.__episode = 1
    # number of episodes a product is served until the current episode
    # (exclusive)
    self.__serving_episodes = np.zeros(self.product_num() + 1)
    # number of times the customer chooses a product until the current time
    # (exclusive)
    self.__customer_choices = np.zeros(self.product_num() + 1)
    self.__last_actions = None
    self.__last_feedback = None

  def UCB(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of preference parameters
    """
    # unbiased estimate of preference parameters
    unbiased_est = self.__customer_choices / self.__serving_episodes
    # temperary result
    tmp_result = 48 * np.log(np.sqrt(self.product_num()) * self.__episode +
                             1) / self.__serving_episodes
    ucb = unbiased_est + np.sqrt(unbiased_est * tmp_result) + tmp_result
    ucb[np.isnan(ucb)] = 1
    ucb = np.minimum(ucb, 1)
    return ucb

  def actions(self, context=None) -> List[Tuple[List[int], int]]:
    """
    Returns:
      assortments to serve
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
    else:
      # check if last observation is not a non-purchase
      if self.__last_feedback and self.__last_feedback[0][1][0] != 0:
        return self.__last_actions
      # When a non-purchase observation happens, a new episode is started and
      # a new assortment to be served is calculated
      self.reward.set_preference_params(self.UCB())
      # calculate assortment with the maximum reward using optimistic
      # preference parameters
      _, best_assortment = search_best_assortment(
          reward=self.reward, card_limit=self.card_limit(
          )) if not self.__use_local_search else local_search_best_assortment(
              reward=self.reward,
              search_times=self.__local_search_times,
              card_limit=self.card_limit(),
              init_assortment=(
                  self.__last_actions[0][0] if self.__last_actions else None))
      self.__last_actions = [(best_assortment, 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__customer_choices[feedback[0][1][0]] += 1
    self.__last_feedback = feedback
    self.__time += 1
    if feedback[0][1][0] == 0:
      for product_id in self.__last_actions[0][0]:
        self.__serving_episodes[product_id] += 1
      self.__episode += 1
