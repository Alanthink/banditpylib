from typing import Dict

import math
import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.learners import argmax_or_min_tuple
from .utils import OrdinaryFBBAILearner


class SR(OrdinaryFBBAILearner):
  """Successive rejects policy :cite:`audibert2010best`

  Eliminate one arm in each round.
  """
  def __init__(self, arm_num: int, budget: int, name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    # calculate bar_log_K
    self.__bar_log_K = 0.5 + sum([1 / i for i in range(2, self.arm_num() + 1)])
    if (budget - arm_num) < arm_num * self.__bar_log_K:
      raise Exception('Budget is too small.')

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'sr'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    # calculate pulls assigned to each arm per_round
    self.__pulls_per_round = [-1]
    nk = [0]
    for k in range(1, self.arm_num()):
      nk.append(
          math.ceil(1 / self.__bar_log_K * (self.budget() - self.arm_num()) /
                    (self.arm_num() + 1 - k)))
      self.__pulls_per_round.append(nk[k] - nk[k - 1])

    self.__active_arms: Dict[int, PseudoArm] = dict()
    for arm_id in range(self.arm_num()):
      self.__active_arms[arm_id] = PseudoArm()

    self.__budget_left = self.budget()
    self.__best_arm = None
    # current round
    self.__round = 1

  def actions(self, context=None) -> Actions:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context

    actions = Actions()

    if self.__round < self.arm_num():
      if self.__round < self.arm_num() - 1:
        for arm_id in self.__active_arms:
          arm_pulls_pair = actions.arm_pulls_pairs.add()
          arm_pulls_pair.arm.id = arm_id
          arm_pulls_pair.pulls = self.__pulls_per_round[self.__round]
      else:
        # Use up the remaining budget when there are only two arms left
        pulls = [self.__budget_left // 2]
        pulls.append(self.__budget_left - pulls[0])
        for i in range(2):
          arm_pulls_pair = actions.arm_pulls_pairs.add()
          arm_pulls_pair.arm.id = list(self.__active_arms.keys())[i]
          arm_pulls_pair.pulls = pulls[i]
    return actions

  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    for arm_rewards_pair in feedback.arm_rewards_pairs:
      self.__active_arms[arm_rewards_pair.arm.id].update(
          np.array(arm_rewards_pair.rewards))
      self.__budget_left -= len(arm_rewards_pair.rewards)

    # Eliminate the arm with the smallest mean reward
    arm_id_to_remove = argmax_or_min_tuple(
        [(self.__active_arms[arm_id].em_mean, arm_id)
         for arm_id in self.__active_arms],
        find_min=True)
    del self.__active_arms[arm_id_to_remove]

    if self.__round == self.arm_num() - 1:
      self.__best_arm = list(self.__active_arms.keys())[0]
    self.__round += 1

  def best_arm(self) -> int:
    """
    Returns:
      best arm identified by the learner
    """
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm
