from typing import List, Tuple, Optional

import math
import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryFCBAILearner


class LilUCBHeuristic(OrdinaryFCBAILearner):
  """lilUCB heuristic policy :cite:`jamieson2014lil`"""
  def __init__(self, arm_num: int, confidence: float, name: str = None):
    """
    Args:
      arm_num: number of arms
      confidence: confidence level. It should be within (0, 1).
      name: alias name
    """
    super().__init__(arm_num=arm_num, confidence=confidence, name=name)

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'lilUCB_heur'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # parameters suggested by the paper
    self.__beta = 0.5
    self.__a = 1 + 10 / self.arm_num()
    self.__eps = 0
    self.__delta = (1 - self.confidence()) / 5
    # total number of pulls used
    self.__total_pulls = 0
    self.__stage = 'initialization'

  def confidence_radius(self, pulls: int) -> float:
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

  def ucb(self) -> np.ndarray:
    """
    Returns:
      upper confidence bound
    """
    return np.array([
        pseudo_arm.em_mean + self.confidence_radius(pseudo_arm.total_pulls())
        for pseudo_arm in self.__pseudo_arms
    ])

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    if self.__stage == 'initialization':
      self.__last_actions = [(arm_id, 1) for arm_id in range(self.arm_num())]
    else:
      # self.__stage == 'main'
      for pseudo_arm in self.__pseudo_arms:
        if pseudo_arm.total_pulls() >= (
            1 + self.__a * (self.__total_pulls - pseudo_arm.total_pulls())):
          return None
      self.__last_actions = [(np.argmax(self.ucb()), 1)]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    for (ind, (rewards, _)) in enumerate(feedback):
      self.__pseudo_arms[self.__last_actions[ind][0]].update(rewards)
      self.__total_pulls += len(rewards)
    if self.__stage == 'initialization':
      self.__stage = 'main'

  def best_arm(self) -> int:
    """
    Returns:
      best arm identified by the learner

    .. todo::
      Randomize the output when there are multiple arms with the same empirical
      mean.
    """
    return max([(arm_id, pseudo_arm.total_pulls())
                for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)],
               key=lambda x: x[1])[0]
