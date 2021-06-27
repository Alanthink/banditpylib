from typing import Optional, Dict

import math
import numpy as np

from banditpylib import argmax_or_min_tuple
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABFixedBudgetBAILearner


class SR(MABFixedBudgetBAILearner):
  """Successive rejects policy :cite:`audibert2010best`

  Eliminate one arm in each round.

  :param int arm_num: number of arms
  :param int budget: total number of pulls
  :param Optional[str] name: alias name
  """
  def __init__(self, arm_num: int, budget: int, name: Optional[str] = None):
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    # calculate bar_log_K
    self.__bar_log_K = 0.5 + sum([1 / i for i in range(2, self.arm_num + 1)])
    if (budget - arm_num) < arm_num * self.__bar_log_K:
      raise Exception('Budget is expected at least %d. Got %d.' %
                      (arm_num * self.__bar_log_K + arm_num, budget))

  def _name(self) -> str:
    return 'sr'

  def reset(self):
    # Calculate pulls assigned to each arm per round
    self.__pulls_per_round = [-1]
    nk = [0]
    for k in range(1, self.arm_num):
      nk.append(
          math.ceil(1 / self.__bar_log_K * (self.budget - self.arm_num) /
                    (self.arm_num + 1 - k)))
      self.__pulls_per_round.append(nk[k] - nk[k - 1])

    self.__active_arms: Dict[int, PseudoArm] = dict()
    for arm_id in range(self.arm_num):
      self.__active_arms[arm_id] = PseudoArm()

    self.__budget_left = self.budget
    self.__best_arm = None
    # Current round
    self.__round = 1

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()

    if self.__round < self.arm_num:
      if self.__round < self.arm_num - 1:
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
    for arm_feedback in feedback.arm_feedbacks:
      self.__active_arms[arm_feedback.arm.id].update(
          np.array(arm_feedback.rewards))
      self.__budget_left -= len(arm_feedback.rewards)

    # Eliminate the arm with the smallest mean reward
    arm_id_to_remove = argmax_or_min_tuple(
        [(self.__active_arms[arm_id].em_mean, arm_id)
         for arm_id in self.__active_arms],
        find_min=True)
    del self.__active_arms[arm_id_to_remove]

    if self.__round == self.arm_num - 1:
      self.__best_arm = list(self.__active_arms.keys())[0]
    self.__round += 1

  @property
  def best_arm(self) -> int:
    if self.__best_arm is None:
      raise Exception('I don\'t have an answer yet.')
    return self.__best_arm
