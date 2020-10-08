from typing import List, Tuple, Optional

import math
import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryFCBAILearner


class ExpGap(OrdinaryFCBAILearner):
  """Exponential-gap elimination policy :cite:`karnin2013almost`"""
  def __init__(self,
               arm_num: int,
               confidence: float,
               name: str = None,
               threshold: int = 3):
    """
    Args:
      arm_num: number of arms
      confidence: confidence level. It should be within (0, 1).
      name: alias name
      threshold: do uniform sampling when the active arms are no greater than
        the threshold within median elimination
    """
    super().__init__(arm_num=arm_num, confidence=confidence, name=name)
    if threshold <= 2:
      raise Exception('Thredhold %d is less than 3!' % threshold)
    self.__threshold = threshold

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'exp_gap'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__active_arms = set(range(self.arm_num()))
    self.__best_arm = None
    # current round
    self.__round = 1
    self.__stage = 'main_loop'
    # main loop variables
    self.__eps_r = 0.125
    self.__log_delta_r = math.log((1 - self.confidence()) / 50)

  @property
  def stage(self) -> str:
    """stage of the learner"""
    return self.__stage

  def median_elimination(self) -> List[Tuple[int, int]]:
    """
    Returns:
      arms to pull in median elimination
    """
    self.__me_pseudo_arms = [(arm_id, PseudoArm())
                             for arm_id in self.__me_active_arms]
    if len(self.__me_active_arms) <= self.__threshold:
      # uniform sampling
      pulls = math.ceil(
          0.5 / (self.__me_eps_left**2) *
          (math.log(2 / self.__me_delta_left / len(self.__me_active_arms))))
    else:
      pulls = math.ceil(4 / (self.__me_eps_ell**2) *
                        (math.log(3) - self.__me_log_delta_ell))
    actions = [(arm_id, pulls) for arm_id in self.__me_active_arms]
    return actions

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    if len(self.__active_arms) == 1:
      self.__last_actions = None
    elif self.__stage == 'main_loop':
      self.__pseudo_arms = [(arm_id, PseudoArm())
                            for arm_id in self.__active_arms]
      pulls = math.ceil(2 / (self.__eps_r**2) *
                        (math.log(2) - self.__log_delta_r))
      self.__last_actions = [(arm_id, pulls) for arm_id in self.__active_arms]
    else:
      # self.__stage == 'median_elimination'
      self.__last_actions = self.median_elimination()

    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    if self.__stage == 'main_loop':
      for (ind, (rewards, _)) in enumerate(feedback):
        self.__pseudo_arms[ind][1].update(rewards)
      # initialization of median elimination
      self.__stage = 'median_elimination'
      self.__me_ell = 1
      self.__me_eps_ell = self.__eps_r / 8
      self.__me_log_delta_ell = self.__log_delta_r - math.log(2)
      self.__me_eps_left = self.__eps_r / 2
      self.__me_delta_left = math.exp(self.__log_delta_r)
      self.__me_active_arms = list(self.__active_arms)
    elif self.__stage == 'median_elimination':
      for (ind, (rewards, _)) in enumerate(feedback):
        self.__me_pseudo_arms[ind][1].update(rewards)
      if len(self.__me_active_arms) > self.__threshold:
        median = np.median(
            np.array([
                pseudo_arm.em_mean
                for (arm_id, pseudo_arm) in self.__me_pseudo_arms
            ]))
        self.__me_active_arms = [
            arm_id for (arm_id, pseudo_arm) in self.__me_pseudo_arms
            if pseudo_arm.em_mean >= median
        ]
        self.__me_eps_left *= 0.75
        self.__me_delta_left *= 0.5
        self.__me_eps_ell *= 0.75
        self.__me_log_delta_ell -= math.log(2)
        self.__me_ell += 1
      else:
        eps_best_pseudo_arm = max(self.__me_pseudo_arms,
                                  key=lambda x: x[1].em_mean)[1]
        # second half of 'main_loop'
        # use estimated epsilon-best-arm to do elimination
        self.__active_arms = set([
            arm_id for (arm_id, pseudo_arm) in self.__pseudo_arms
            if pseudo_arm.em_mean >= eps_best_pseudo_arm.em_mean - self.__eps_r
        ])
        if len(self.__active_arms) == 1:
          self.__best_arm = list(self.__active_arms)[0]
        self.__stage = 'main_loop'
        self.__round += 1
        self.__eps_r /= 2
        self.__log_delta_r = math.log(
            (1 - self.confidence()) / 50) - 3 * math.log(self.__round)

  def best_arm(self) -> int:
    """
    Returns:
      best arm identified by the learner

    .. todo::
      Randomize the output when there are multiple arms with the same empirical
      mean.
    """
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm
