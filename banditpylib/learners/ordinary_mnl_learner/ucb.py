from typing import Optional

import numpy as np

from banditpylib.bandits import search_best_assortment, Reward, \
    local_search_best_assortment
from banditpylib.data_pb2 import Actions, Feedback
from .utils import OrdinaryMNLLearner


class UCB(OrdinaryMNLLearner):
  """UCB policy :cite:`DBLP:journals/ior/AgrawalAGZ19`

  :param np.ndarray revenues: product revenues
  :param Reward reward: reward the learner wants to maximize
  :param int card_limit: cardinality constraint
  :param bool use_local_search: whether to use local search for searching the
    best assortment
  :param int random_neighbors: number of random neighbors to look up if local
    search is enabled
  :param Optional[str] name: alias name
  """
  def __init__(
      self,
      revenues: np.ndarray,
      reward: Reward,
      card_limit: int = np.inf,  # type: ignore
      use_local_search: bool = False,
      random_neighbors: int = 10,
      name: Optional[str] = None):
    super().__init__(revenues=revenues,
                     reward=reward,
                     card_limit=card_limit,
                     use_local_search=use_local_search,
                     random_neighbors=random_neighbors,
                     name=name)

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'risk_aware_ucb'

  def reset(self):
    # Current time step
    self.__time = 1
    # Current episode
    self.__episode = 1
    # Number of episodes a product is served until the current episode
    # (exclusive)
    self.__serving_episodes = np.zeros(self.product_num + 1)
    # Number of times the customer chooses a product until the current time
    # (exclusive)
    self.__customer_choices = np.zeros(self.product_num + 1)
    self.__last_actions = None
    self.__last_customer_feedback = None

  def __UCB(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of preference parameters
    """
    # Unbiased estimate of preference parameters
    unbiased_est = self.__customer_choices / self.__serving_episodes
    # Temperary result
    tmp_result = 48 * np.log(np.sqrt(self.product_num) * self.__episode +
                             1) / self.__serving_episodes
    ucb = unbiased_est + np.sqrt(unbiased_est * tmp_result) + tmp_result
    ucb[np.isnan(ucb)] = 1
    ucb = np.minimum(ucb, 1)
    return ucb

  def actions(self, context=None) -> Actions:
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()

    # Check if last observation is a purchase
    if self.__last_customer_feedback and self.__last_customer_feedback != 0:
      return self.__last_actions
    # When a non-purchase observation happens, a new episode is started and
    # a new assortment to be served is calculated
    self.reward.set_preference_params(self.__UCB())
    # Calculate assortment with the maximum reward using optimistic
    # preference parameters
    if self.use_local_search:
      _, best_assortment = local_search_best_assortment(
          reward=self.reward,
          random_neighbors=self.random_neighbors,
          card_limit=self.card_limit,
          init_assortment=(set(
              self.__last_actions.arm_pulls_pairs[0].arm.set.id)
                           if self.__last_actions else None))
    else:
      _, best_assortment = search_best_assortment(reward=self.reward,
                                                  card_limit=self.card_limit)

    arm_pulls_pair.arm.set.id.extend(list(best_assortment))
    arm_pulls_pair.pulls = 1

    self.__last_actions = actions
    return actions

  def update(self, feedback: Feedback):
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    self.__customer_choices[arm_rewards_pair.customer_feedbacks[0]] += 1

    self.__last_customer_feedback = arm_rewards_pair.customer_feedbacks[0]
    self.__time += 1
    if arm_rewards_pair.customer_feedbacks[0] == 0:
      for product_id in self.__last_actions.arm_pulls_pairs[0].arm.set.id:
        self.__serving_episodes[product_id] += 1
      self.__episode += 1
