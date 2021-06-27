from typing import Optional

from absl import logging
import numpy as np

from banditpylib.bandits import search_best_assortment, Reward, \
    local_search_best_assortment
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MNLBanditLearner


class ThompsonSampling(MNLBanditLearner):
  """Thompson sampling policy :cite:`DBLP:conf/colt/AgrawalAGZ17`

  :param np.ndarray revenues: product revenues
  :param int horizon: total number of time steps
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
      horizon: int,
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
    if horizon < self.product_num:
      logging.warning('Horizon %d is less than number of products %d!' % \
          (horizon, self.product_num))
    self.__horizon = horizon

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'thompson_sampling'

  def reset(self):
    # Current time step
    self.__time = 1
    # Current episode
    self.__episode = 1
    # Number of episodes a product is served until the current episode
    # (exclusive)
    self.__serving_episodes = np.zeros(self.product_num + 1)
    # Number of times a product is picked until the current time (exclusive)
    self.__customer_choices = np.zeros(self.product_num + 1)
    self.__last_actions = None
    self.__last_customer_feedback = None
    # Flag to denote whether the initial warm start stage has finished
    self.__done_warm_start = False
    # Next product to try in the warm start stage
    self.__next_product_in_warm_start = 1

  def __warm_start(self) -> Actions:
    """Initial warm start stage

    Returns:
      assortments to serve in the warm start stage
    """
    # Check if last observation is a purchase
    if self.__last_customer_feedback and self.__last_customer_feedback != 0:
      # Continue serving the same assortment
      return self.__last_actions

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()
    arm_pulls_pair.arm.set.id.append(self.__next_product_in_warm_start)
    arm_pulls_pair.pulls = 1
    self.__next_product_in_warm_start += 1
    return actions

  def __within_warm_start(self) -> bool:
    """
    Returns:
      `True` if the learner is still in warm start stage
    """
    return not self.__done_warm_start

  def __correlated_sampling(self) -> np.ndarray:
    """
    Returns:
      correlated sampling of preference parameters
    """
    theta = np.max(np.random.normal(0, 1, self.card_limit))
    # Unbiased estimate of preference parameters
    unbiased_est = self.__customer_choices / self.__serving_episodes
    sampled_preference_params = unbiased_est + theta * (
        np.sqrt(50 * unbiased_est *
                (unbiased_est + 1) / self.__serving_episodes) +
        75 * np.sqrt(np.log(self.__horizon * self.card_limit)) /
        self.__serving_episodes)
    sampled_preference_params[0] = 1
    sampled_preference_params = np.minimum(sampled_preference_params, 1)
    return sampled_preference_params

  def actions(self, context: Context) -> Actions:
    del context

    actions: Actions

    # Check if still in warm start stage
    if self.__within_warm_start():
      actions = self.__warm_start()
    else:
      actions = Actions()
      arm_pulls_pair = actions.arm_pulls_pairs.add()

      # Check if last observation is a purchase
      if self.__last_customer_feedback and self.__last_customer_feedback != 0:
        # Continue serving the same assortment
        return self.__last_actions

      # When a non-purchase observation happens, a new episode is started. Also
      # a new assortment to be served using new estimate of preference
      # parameters is generated.
      # Set preference parameters generated by thompson sampling
      self.reward.set_preference_params(self.__correlated_sampling())
      # Calculate best assortment using the generated preference parameters
      if self.use_local_search:
        # Initial assortment to start for local search
        if self.__last_actions is not None:
          init_assortment = set(
              self.__last_actions.arm_pulls_pairs[0].arm.set.id)
        else:
          init_assortment = None
        _, best_assortment = local_search_best_assortment(
            reward=self.reward,
            random_neighbors=self.random_neighbors,
            card_limit=self.card_limit,
            init_assortment=init_assortment)
      else:
        _, best_assortment = search_best_assortment(reward=self.reward,
                                                    card_limit=self.card_limit)

      arm_pulls_pair.arm.set.id.extend(list(best_assortment))
      arm_pulls_pair.pulls = 1

      self.__first_step_after_warm_start = False

    self.__last_actions = actions
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]
    self.__customer_choices[arm_feedback.customer_feedbacks[0]] += 1

    # No purchase is observed
    if arm_feedback.customer_feedbacks[0] == 0:
      for product_id in self.__last_actions.arm_pulls_pairs[0].arm.set.id:
        self.__serving_episodes[product_id] += 1
      # Check if it is the end of initial warm start stage
      if not self.__done_warm_start and \
          self.__next_product_in_warm_start > self.product_num:
        self.__done_warm_start = True
        self.__last_actions = None
      self.__episode += 1
    self.__last_customer_feedback = arm_feedback.customer_feedbacks[0]
    self.__time += 1
