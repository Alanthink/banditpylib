import copy
from typing import Set

from absl import logging

import numpy as np

from banditpylib.data_pb2 import Actions, Feedback, ArmPullsPair, ArmRewardsPair
from banditpylib.learners import Goal, MaximizeTotalRewards
from .ordinary_mnl_bandit_utils import Reward, MeanReward, \
    search_best_assortment
from .utils import Bandit


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

  :param np.ndarray reference_params: preference parameters (product 0 should
    be included)
  :param np.ndarray revenue: revenue of products (product 0 should be included)
  :param int card_limit: cardinality constraint of an assortment meaning the
    total number of products provided at a time is no greater than this number
  :param Reward reward: reward the learner wants to maximize. The default goal
    is mean of rewards
  :param bool zero_best_reward: whether to set the reward of the best
    assortment to 0. This is useful when data is too large to compute the best
    assortment. When best reward is set to zero, the regret equals to the minus
    total revenue.
  """
  def __init__(
      self,
      preference_params: np.ndarray,
      revenues: np.ndarray,
      card_limit: int = np.inf,  # type: ignore
      reward: Reward = None,
      zero_best_reward: bool = False):
    if len(preference_params) != len(revenues):
      raise ValueError(
          'Number of preference parameters %d is expected equal to number of '
          'revenues %d.' % (len(preference_params), len(revenues)))
    for (i, param) in enumerate(preference_params):
      if param > 1 or param < 0:
        raise ValueError('The %d-th preference parameter is '
                         'expected within [0, 1].' % i)
    if preference_params[0] != 1:
      raise ValueError(
          'The preference parameter of product 0 is expected 1. Got %.2f.' %
          preference_params[0])
    for (i, revenue) in enumerate(revenues):
      if i > 0 and revenue <= 0:
        raise ValueError('The %d-th revenue is expected greater than 0.' % i)
    if revenues[0] != 0:
      raise ValueError('The revenue of product 0 is expected 0. Got %.2f.' %
                       revenues[0])

    self.__preference_params = preference_params
    self.__revenues = revenues
    # Product 0 is reserved for non-purchase
    self.__product_num = len(self.__preference_params) - 1
    if self.__product_num == 0:
      raise ValueError('Number of products is expected at least 1. Got 0.')
    if card_limit < 1:
      raise ValueError('Cardinality limit is expected at least 1. Got %d.' %
                       card_limit)
    self.__card_limit = min(card_limit, self.__product_num)

    # Maximizing the rewards is the default goal
    self.__reward = MeanReward() if reward is None else copy.deepcopy(reward)
    self.__reward.set_preference_params(self.__preference_params)
    self.__reward.set_revenues(self.__revenues)

    self.__best_assort: Set[int]

    if zero_best_reward:
      self.__best_reward, self.__best_assort = 0.0, set()
      logging.warning(
          'Best reward is set to zero. Now the regret equals to the'
          ' minus total revenue.')
    else:
      # Compute the best assortment
      self.__best_reward, self.__best_assort = search_best_assortment(
          reward=self.__reward, card_limit=self.__card_limit)
      logging.info('Assortment %s has best reward %.2f.',
                   sorted(list(self.__best_assort)), self.__best_reward)

  @property
  def name(self) -> str:
    return 'ordinary_mnl_bandit'

  def _take_action(self, arm_pulls_pair: ArmPullsPair) -> ArmRewardsPair:
    """Serve one assortment

    Args:
      arm_pulls_pair: assortment and number of serving times

    Returns:
      feedbacks of the customer
    """
    assortment = set(arm_pulls_pair.arm.set.id)
    times = arm_pulls_pair.pulls

    if not assortment:
      raise Exception('Empty assortment!')
    for product_id in assortment:
      if product_id < 1 or product_id > self.__product_num:
        raise Exception('Product id %d is out of range [1, %d]!' %
                        (product_id, self.__product_num))
    if len(assortment) > self.__card_limit:
      raise Exception('Assortment %s has products more than cardinality'
                      ' constraint %d!' %
                      (sorted(list(assortment)), self.__card_limit))

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

    arm_rewards_pair = ArmRewardsPair()
    arm_rewards_pair.arm.set.id.extend(list(assortment))
    arm_rewards_pair.rewards.extend(
        np.array([self.__revenues[choice] for choice in choices]))
    arm_rewards_pair.customer_feedbacks.extend(choices)

    # Update regret
    self.__regret += (self.__best_reward -
                      self.__reward.calc(assortment)) * times

    return arm_rewards_pair

  def feed(self, actions: Actions) -> Feedback:
    feedback = Feedback()
    for arm_pulls_pair in actions.arm_pulls_pairs:
      arm_rewards_pair = self._take_action(arm_pulls_pair=arm_pulls_pair)
      if arm_rewards_pair.rewards:
        feedback.arm_rewards_pairs.append(arm_rewards_pair)
    return feedback

  def reset(self):
    self.__regret = 0.0

  @property
  def context(self):
    return None

  @property
  def revenues(self) -> np.ndarray:
    """Revenues of products (product 0 is included, which is always 0.0)"""
    return self.__revenues

  @property
  def product_num(self) -> int:
    """Number of products (not including product 0)"""
    return self.__product_num

  @property
  def card_limit(self) -> float:
    """Cardinality limit"""
    return self.__card_limit

  def regret(self, goal: Goal) -> float:
    if isinstance(goal, MaximizeTotalRewards):
      return self.__regret
    raise Exception('Goal %s is not supported!' % goal.name)
