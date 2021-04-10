from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.learners import Goal, AllCorrect
from .utils import ThresBanditLearner


class APT(ThresBanditLearner):
  """Anytime Parameter-free Thresholding algorithm
  :cite:`DBLP:conf/icml/LocatelliGC16`
  """
  def __init__(self,
               arm_num: int,
               budget: int,
               theta: float,
               eps: float,
               name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      theta: threshold
      eps: radius of indifferent zone
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    self.__theta = theta
    self.__eps = eps

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'apt'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def __metrics(self) -> np.ndarray:
    """
    Returns:
      metrics of apt for each arm
    """
    metrics = np.array([
        np.sqrt(arm.total_pulls()) *
        (np.abs(arm.em_mean - self.__theta) + self.__eps)
        for arm in self.__pseudo_arms
    ])
    return metrics

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the thresholding bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__time > self.budget():
      self.__last_actions = None
    elif self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    else:
      self.__last_actions = [(np.argmin(self.__metrics()), 1)]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        `self.__last_actions`
    """
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

  @property
  def goal(self) -> Goal:
    answers = [
        1 if arm.em_mean >= self.__theta else 0 for arm in self.__pseudo_arms
    ]
    return AllCorrect(answers=answers)
