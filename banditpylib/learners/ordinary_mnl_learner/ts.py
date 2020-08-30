from typing import List, Tuple

from absl import logging

import numpy as np

from banditpylib.bandits import search_best_assortment, Reward, \
    local_search_best_assortment
from .utils import OrdinaryMNLLearner


class ThompsonSampling(OrdinaryMNLLearner):
  """Thompson sampling policy :cite:`DBLP:conf/colt/AgrawalAGZ17`"""
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
      revenues: product revenues
      horizon: total number of time steps
      reward: reward the learner wants to maximize
      card_limit: cardinality constraint
      name: alias name for the learner
      use_local_search: whether to use local search for searching the best \
      assortment
      local_search_times: number of local searches if local search is used
    """
    self.__name = name if name else 'thompson_sampling'
    super().__init__(revenues, horizon, reward, card_limit)
    self.__use_local_search = use_local_search
    if local_search_times < 10:
      logging.fatal('Times of local search %d is less than 10!' %
                    local_search_times)
    self.__local_search_times = local_search_times

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
    self.__serving_episodes = np.zeros(self.product_num() + 1)
    # number of times the customer chooses a product until the current time
    # (exclusive)
    self.__customer_choices = np.zeros(self.product_num() + 1)
    self.__last_actions = None
    self.__last_feedback = None
    # flag to denote whether the initial warm start stage has finished
    self.__done_warm_start = False
    # next product to try in the warm start stage
    self.__next_product_in_warm_start = 1

  def warm_start(self) -> List[Tuple[List[int], int]]:
    """Initial warm start stage

    Return:
      assortment to serve
    """
    # check if last observation is a purchase
    if self.__last_feedback and self.__last_feedback[0][1][0] != 0:
      # continue serving the same assortment
      return self.__last_actions
    self.__last_actions = [([self.__next_product_in_warm_start], 1)]
    self.__next_product_in_warm_start += 1
    return self.__last_actions

  def within_warm_start(self) -> bool:
    """
    Return:
      if the learner is still in warm start stage
    """
    return not self.__done_warm_start

  def sample_preference_params(self) -> np.ndarray:
    """
    Return:
      one time sampling of preference parameters
    """
    theta = np.random.normal(0, 1, self.product_num() + 1)
    # unbiased estimate of preference parameters
    unbiased_est = self.__customer_choices / self.__serving_episodes
    sampled_preference_params = unbiased_est + theta * (
        np.sqrt(50 * unbiased_est *
                (unbiased_est + 1) / self.__serving_episodes) +
        75 * np.sqrt(np.log(self.horizon() * self.card_limit())) /
        self.__serving_episodes)
    return sampled_preference_params

  def thompson_sampling(self) -> np.ndarray:
    """
    Return:
      virtual preference parameters sampled using thompson sampling
    """
    virtual_preference_params = np.zeros(self.product_num() + 1)
    for _ in range(self.card_limit()):
      virtual_preference_params = np.maximum(virtual_preference_params,
                                             self.sample_preference_params())
    virtual_preference_params[np.isnan(virtual_preference_params)] = 1
    virtual_preference_params = np.minimum(virtual_preference_params, 1)
    return virtual_preference_params

  def actions(self, context=None) -> List[Tuple[List[int], int]]:
    """
    Return:
      assortments to serve
    """
    del context
    # check if the learner should stop the game
    if self.__time > self.horizon():
      self.__last_actions = None
    # check if warm start has not finished
    elif self.within_warm_start():
      self.__last_actions = self.warm_start()
    else:
      # check if last observation is a purchase
      if self.__last_feedback and self.__last_feedback[0][1][0] != 0:
        # continue serving the same assortment
        return self.__last_actions

      # When a non-purchase observation happens, a new episode is started and
      # a new assortment to be served is calculated.
      self.reward.set_preference_params(self.thompson_sampling())
      # calculate best assortment using the virtual preference parameters
      # sampled from a posterior distribution
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
    # a non-purchase is observed
    if feedback[0][1][0] == 0:
      for product_id in self.__last_actions[0][0]:
        self.__serving_episodes[product_id] += 1
      # check if it is the end of initial warm start stage
      if self.__next_product_in_warm_start > self.product_num():
        self.__done_warm_start = True
      self.__episode += 1
    self.__last_feedback = feedback
    self.__time += 1
