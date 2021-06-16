from typing import Optional

import math
import numpy as np

from banditpylib import argmax_or_min_tuple
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import OrdinaryFCBAILearner


class LilUCBHeuristic(OrdinaryFCBAILearner):
  """LilUCB heuristic policy :cite:`jamieson2014lil`

  :param int arm_num: number of arms
  :param float confidence: confidence level. It should be within (0, 1). The
    algorithm should output the best arm with probability at least this value.
  :param Optional[str] name: alias name
  """
  def __init__(self,
               arm_num: int,
               confidence: float,
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, confidence=confidence, name=name)

  def _name(self) -> str:
    return 'lilUCB_heur'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    # Parameters suggested by the paper
    self.__beta = 0.5
    self.__a = 1 + 10 / self.arm_num
    self.__eps = 0
    self.__delta = (1 - self.confidence) / 5
    # Total number of pulls used
    self.__total_pulls = 0
    self.__stage = 'initialization'

  def __confidence_radius(self, pulls: int) -> float:
    """
    Args:
      pulls: number of pulls

    Returns:
      confidence radius
    """
    if (1 + self.__eps) * pulls == 1:
      return math.inf
    return (1 + self.__beta) * (1 + math.sqrt(self.__eps)) * math.sqrt(
        2 * (1 + self.__eps) *
        math.log(math.log((1 + self.__eps) * pulls) / self.__delta) / pulls)

  def __ucb(self) -> np.ndarray:
    """
    Returns:
      upper confidence bound
    """
    return np.array([
        pseudo_arm.em_mean + self.__confidence_radius(pseudo_arm.total_pulls)
        for pseudo_arm in self.__pseudo_arms
    ])

  def actions(self, context: Context) -> Actions:
    if self.__stage == 'initialization':
      actions = Actions()
      for arm_id in range(self.arm_num):
        arm_pulls_pair = actions.arm_pulls_pairs.add()
        arm_pulls_pair.arm.id = arm_id
        arm_pulls_pair.pulls = 1
      return actions

    # self.__stage == 'main'
    actions = Actions()

    for pseudo_arm in self.__pseudo_arms:
      if pseudo_arm.total_pulls >= (
          1 + self.__a * (self.__total_pulls - pseudo_arm.total_pulls)):
        return actions

    arm_pulls_pair = actions.arm_pulls_pairs.add()
    arm_pulls_pair.arm.id = int(np.argmax(self.__ucb()))
    arm_pulls_pair.pulls = 1

    return actions

  def update(self, feedback: Feedback):
    for arm_rewards_pair in feedback.arm_rewards_pairs:
      self.__pseudo_arms[arm_rewards_pair.arm.id].update(
          np.array(arm_rewards_pair.rewards))
      self.__total_pulls += len(arm_rewards_pair.rewards)

    if self.__stage == 'initialization':
      self.__stage = 'main'

  @property
  def best_arm(self) -> int:
    return argmax_or_min_tuple([
        (pseudo_arm.total_pulls, arm_id)
        for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
    ])
