from typing import List, Tuple, Optional

import math
import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.learners import argmax_tuple
from .utils import OrdinaryFBBAILearner


class SH(OrdinaryFBBAILearner):
  """Sequential halving policy :cite:`karnin2013almost`"""
  def __init__(self,
               arm_num: int,
               budget: int,
               threshold: int = 2,
               name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      threshold: do uniform sampling when the number of arms left is no greater
        than this number
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    if threshold < 2:
      raise Exception('Thredhold %d is less than 2!' % threshold)
    self.__threshold = threshold
    if budget < (arm_num * math.ceil(math.log(self.arm_num(), 2))):
      raise Exception('Budget is too small.')

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'sh'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    self.__active_arms = list(range(self.arm_num()))
    self.__budget_left = self.budget()
    self.__best_arm = None
    self.__total_rounds = math.ceil(math.log(self.arm_num(), 2))
    # current round
    self.__round = 1
    self.__last_round = False

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__last_round:
      self.__last_actions = None
    elif len(self.__active_arms) <= self.__threshold:
      # uniform sampling
      pulls = np.random.multinomial(self.__budget_left,
                                    np.ones(len(self.__active_arms)) /
                                    len(self.__active_arms),
                                    size=1)[0]
      self.__last_actions = [(self.__active_arms[i], pulls[i])
                             for i in range(len(self.__active_arms))]
      self.__last_round = True
    else:
      # pulls assigned to each arm
      pulls = math.floor(self.budget() /
                         (len(self.__active_arms) * self.__total_rounds))
      self.__last_actions = [(arm_id, pulls)
                             for arm_id in self.__active_arms]
    return self.__last_actions

  def update(self, feedback: List[Tuple[Optional[np.ndarray], None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    for (ind, (rewards, _)) in enumerate(feedback):
      self.__pseudo_arms[self.__last_actions[ind][0]].update(rewards)
      self.__budget_left -= len(rewards)
    if self.__last_round:
      self.__best_arm = argmax_tuple([
          (self.__pseudo_arms[arm_id].em_mean,
           arm_id) for arm_id in self.__active_arms])
    else:
      # remove half of the arms with the worst empirical means
      sorted_active_arms = sorted(self.__active_arms,
                                  key=lambda x: self.__pseudo_arms[x].em_mean,
                                  reverse=True)
      self.__active_arms = sorted_active_arms[:math.
                                              ceil(len(self.__active_arms) / 2)]
    self.__round += 1

  def best_arm(self) -> int:
    """
    Returns:
      best arm identified by the learner
    """
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm
