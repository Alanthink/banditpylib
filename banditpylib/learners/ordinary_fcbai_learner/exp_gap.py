from typing import Dict

import math
import numpy as np

from banditpylib import argmax_or_min_tuple
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from .utils import OrdinaryFCBAILearner


class ExpGap(OrdinaryFCBAILearner):
  """Exponential-gap elimination policy :cite:`karnin2013almost`

  :param int arm_num: number of arms
  :param float confidence confidence: confidence level. It should be within
    (0, 1). The algorithm should output the best arm with probability at least
    this value.
  :param int threshold: do uniform sampling when the active arms are no greater
    than the threshold within median elimination
  :param str name: alias name
  """
  def __init__(self,
               arm_num: int,
               confidence: float,
               threshold: int = 2,
               name: str = None):
    super().__init__(arm_num=arm_num, confidence=confidence, name=name)
    if threshold < 2:
      raise Exception('Thredhold %d is less than 2!' % threshold)
    self.__threshold = threshold

  def _name(self) -> str:
    return 'exp_gap'

  def reset(self):
    self.__active_arms: Dict[int, PseudoArm] = dict()
    for arm_id in range(self.arm_num()):
      self.__active_arms[arm_id] = PseudoArm()

    self.__best_arm = None
    # Current round index
    self.__round = 1
    self.__stage = 'main_loop'
    # Main loop variables
    self.__eps_r = 0.125
    self.__log_delta_r = math.log((1 - self.confidence()) / 50)

  @property
  def stage(self) -> str:
    """Stage of the learner"""
    return self.__stage

  def __median_elimination(self) -> Actions:
    """
    Returns:
      arms to pull in median elimination
    """
    actions = Actions()
    for arm_id in self.__me_active_arms:
      self.__me_active_arms[arm_id] = PseudoArm()

    if len(self.__me_active_arms) <= self.__threshold:
      # Uniform sampling
      pulls = math.ceil(
          0.5 / (self.__me_eps_left**2) *
          (math.log(2 / self.__me_delta_left / len(self.__me_active_arms))))
    else:
      pulls = math.ceil(4 / (self.__me_eps_ell**2) *
                        (math.log(3) - self.__me_log_delta_ell))

    for arm_id in self.__me_active_arms:
      arm_pulls_pair = actions.arm_pulls_pairs.add()
      arm_pulls_pair.arm.id = arm_id
      arm_pulls_pair.pulls = pulls
    return actions

  def actions(self, context=None) -> Actions:
    if len(self.__active_arms) == 1:
      return Actions()

    actions: Actions
    if self.__stage == 'main_loop':
      actions = Actions()
      for arm_id in self.__active_arms:
        self.__active_arms[arm_id] = PseudoArm()

      pulls = math.ceil(2 / (self.__eps_r**2) *
                        (math.log(2) - self.__log_delta_r))
      for arm_id in self.__active_arms:
        arm_pulls_pair = actions.arm_pulls_pairs.add()
        arm_pulls_pair.arm.id = arm_id
        arm_pulls_pair.pulls = pulls
    else:
      # self.__stage == 'median_elimination'
      actions = self.__median_elimination()

    return actions

  def update(self, feedback: Feedback):
    if self.__stage == 'main_loop':
      for arm_rewards_pair in feedback.arm_rewards_pairs:
        self.__active_arms[arm_rewards_pair.arm.id].update(
            np.array(arm_rewards_pair.rewards))
      # Initialization of median elimination
      self.__stage = 'median_elimination'
      self.__me_ell = 1
      self.__me_eps_ell = self.__eps_r / 8
      self.__me_log_delta_ell = self.__log_delta_r - math.log(2)
      self.__me_eps_left = self.__eps_r / 2
      self.__me_delta_left = math.exp(self.__log_delta_r)

      self.__me_active_arms = dict()
      for arm_id in self.__active_arms:
        self.__me_active_arms[arm_id] = PseudoArm()

    elif self.__stage == 'median_elimination':
      for arm_rewards_pair in feedback.arm_rewards_pairs:
        self.__me_active_arms[arm_rewards_pair.arm.id].update(
            np.array(arm_rewards_pair.rewards))
      if len(self.__me_active_arms) > self.__threshold:
        median = np.median(
            np.array([
                pseudo_arm.em_mean
                for (arm_id, pseudo_arm) in self.__me_active_arms.items()
            ]))
        for (arm_id, pseudo_arm) in list(self.__me_active_arms.items()):
          if pseudo_arm.em_mean < median:
            del self.__me_active_arms[arm_id]

        self.__me_eps_left *= 0.75
        self.__me_delta_left *= 0.5
        self.__me_eps_ell *= 0.75
        self.__me_log_delta_ell -= math.log(2)
        self.__me_ell += 1
      else:
        # Best arm returned by median elimination
        best_arm_by_me = argmax_or_min_tuple([
            (pseudo_arm.em_mean, arm_id)
            for arm_id, pseudo_arm in self.__me_active_arms.items()
        ])
        # Second half of 'main_loop'
        # Use estimated epsilon-best-arm to do elimination
        for (arm_id, pseudo_arm) in list(self.__active_arms.items()):
          if pseudo_arm.em_mean < self.__active_arms[
              best_arm_by_me].em_mean - self.__eps_r:
            del self.__active_arms[arm_id]

        if len(self.__active_arms) == 1:
          self.__best_arm = list(self.__active_arms.keys())[0]
        self.__stage = 'main_loop'
        self.__round += 1
        self.__eps_r /= 2
        self.__log_delta_r = math.log(
            (1 - self.confidence()) / 50) - 3 * math.log(self.__round)

  def best_arm(self) -> int:
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm
