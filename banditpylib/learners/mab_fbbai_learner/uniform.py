from typing import Optional

import numpy as np

from banditpylib import argmax_or_min
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABFixedBudgetBAILearner


class Uniform(MABFixedBudgetBAILearner):
  """Uniform sampling policy

  Play each arm the same number of times and then output the arm with the
  highest empirical mean.

  :param int arm_num: number of arms
  :param int budget: total number of pulls
  :param Optional[str] name: alias name
  """
  def __init__(self, arm_num: int, budget: int, name: Optional[str] = None):
    super().__init__(arm_num=arm_num, budget=budget, name=name)

  def _name(self) -> str:
    return 'uniform'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    self.__best_arm = None
    self.__stop = False

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()

    if not self.__stop:
      # Make sure each arm is sampled at least once
      pulls = np.random.multinomial(self.budget - self.arm_num,
                                    np.ones(self.arm_num) / self.arm_num,
                                    size=1)[0]
      for arm_id in range(self.arm_num):
        arm_pull = actions.arm_pulls.add()
        arm_pull.arm.id = arm_id
        arm_pull.times = pulls[arm_id] + 1

      self.__stop = True

    return actions

  def update(self, feedback: Feedback):
    for arm_feedback in feedback.arm_feedbacks:
      self.__pseudo_arms[arm_feedback.arm.id].update(
          np.array(arm_feedback.rewards))
    if self.__stop:
      self.__best_arm = argmax_or_min(
          [arm.em_mean for arm in self.__pseudo_arms])

  @property
  def best_arm(self) -> int:
    if self.__best_arm is None:
      raise Exception('I don\'t have an answer yet.')
    return self.__best_arm
