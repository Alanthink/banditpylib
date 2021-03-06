from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import PseudoCategArm
from banditpylib.learners import Goal, AllCorrect
from .utils import MultiThresBanditLearner


class M_LSA(MultiThresBanditLearner):
  def __init__(self,
               arm_num: int,
               budget: int,
               categ_num: int,
               name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      categ_num: The number of categories to each question
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    self.__categ_num = categ_num

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'm_lsa'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_categ_arms = [PseudoCategArm([0]*self.__categ_num) for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def __metrics(self) -> np.ndarray:
    """
    Returns:
      metrics of m_lsa for each arm
    """
    def metric(arm):
      m1 = arm.sorted_freq[-1]
      m2 = arm.sorted_freq[-2]
      total = m1+m2
      return total * (((m1-m2)/total)/2)**2 + 0.5 * np.log(total)

    metrics = [
        metric(arm) for arm in self.__pseudo_categ_arms
    ]
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
    self.__pseudo_categ_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

  @property
  def goal(self) -> Goal:
    answers = [
        arm.top_categ for arm in self.__pseudo_categ_arms
    ]
    return AllCorrect(answers=answers)
