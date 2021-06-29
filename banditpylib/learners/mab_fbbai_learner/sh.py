from typing import Optional, Dict

import math
import numpy as np

from banditpylib import argmax_or_min_tuple
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABFixedBudgetBAILearner


class SH(MABFixedBudgetBAILearner):
  """Sequential halving policy :cite:`karnin2013almost`

  Eliminate half of the remaining arms in each round.

  :param int arm_num: number of arms
  :param int budget: total number of pulls
  :param int threshold: do uniform sampling when the number of arms left is no
    greater than this number
  :param Optional[str] name: alias name
  """
  def __init__(self,
               arm_num: int,
               budget: int,
               threshold: int = 2,
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    if threshold < 2:
      raise ValueError('Thredhold is expected at least 2. Got %d.' % threshold)
    self.__threshold = threshold
    if budget < (arm_num * math.ceil(math.log(self.arm_num, 2))):
      raise ValueError(
          'Budget is expected at least %d. Got %d.' %
          ((arm_num * math.ceil(math.log(self.arm_num, 2))), budget))

  def _name(self) -> str:
    return 'sh'

  def reset(self):
    self.__active_arms: Dict[int, PseudoArm] = dict()
    for arm_id in range(self.arm_num):
      self.__active_arms[arm_id] = PseudoArm()

    self.__budget_left = self.budget
    self.__best_arm = None
    self.__total_rounds = math.ceil(math.log(self.arm_num, 2))
    # Current round
    self.__round = 1
    self.__stop = False

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()

    if self.__stop:
      return actions

    if len(self.__active_arms) <= self.__threshold:
      # Uniform sampling
      pulls = np.random.multinomial(self.__budget_left,
                                    np.ones(len(self.__active_arms)) /
                                    len(self.__active_arms),
                                    size=1)[0]
      i = 0
      for arm_id in self.__active_arms:
        arm_pull = actions.arm_pulls.add()
        arm_pull.arm.id = arm_id
        arm_pull.times = pulls[i]
        i = i + 1
      self.__stop = True
    else:
      # Pulls assigned to each arm
      pulls = math.floor(self.budget /
                         (len(self.__active_arms) * self.__total_rounds))
      for arm_id in self.__active_arms:
        arm_pull = actions.arm_pulls.add()
        arm_pull.arm.id = arm_id
        arm_pull.times = pulls

    return actions

  def update(self, feedback: Feedback):
    for arm_feedback in feedback.arm_feedbacks:
      self.__active_arms[arm_feedback.arm.id].update(
          np.array(arm_feedback.rewards))
      self.__budget_left -= len(arm_feedback.rewards)
    if self.__stop:
      self.__best_arm = argmax_or_min_tuple([
          (self.__active_arms[arm_id].em_mean, arm_id)
          for arm_id in self.__active_arms
      ])
    else:
      # Remove half of the arms with the worst empirical means
      remaining_arms = sorted(
          self.__active_arms.items(), key=lambda x: x[1].em_mean,
          reverse=True)[:math.ceil(len(self.__active_arms) / 2)]
      self.__active_arms = dict((x, PseudoArm()) for x, _ in remaining_arms)
    self.__round += 1

  @property
  def best_arm(self) -> int:
    if self.__best_arm is None:
      raise Exception('I don\'t have an answer yet!')
    return self.__best_arm
