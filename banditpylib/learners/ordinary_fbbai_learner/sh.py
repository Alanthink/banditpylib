from typing import List, Tuple

import math
import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryFBBAILearner


class SH(OrdinaryFBBAILearner):
  """Sequential halving policy :cite:`karnin2013almost`"""
  def __init__(self, arm_num: int, budget: int,
               name: str = None, threshold: int = 3):
    """
    Args:
      arm_num: number of arms
      horizon: total number of time steps
      name: alias name
      threshold: do uniform sampling when the number of arms left is no greater
        than threshold
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    if threshold <= 2:
      raise Exception('Thredhold %d is no greater than 2!' % threshold)
    self.__threshold = threshold

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'sh'

  def reset(self):
    """Learner reset

    Initialization. This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    self.__active_arms = set(range(self.arm_num()))
    self.__budget_left = self.budget()
    self.__best_arm = None
    self.__total_rounds = math.ceil(math.log(self.arm_num(), 2))
    # current round
    self.__round = 1
    self.__last_round = False

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__budget_left == 0:
      self.__last_actions = None
    elif len(self.__active_arms) <= self.__threshold:
      # uniform sampling
      pulls = np.random.multinomial(self.__budget_left,
                                    np.ones(len(self.__active_arms)) /
                                    len(self.__active_arms),
                                    size=1)[0]
      self.__last_actions = [(list(self.__active_arms)[i], pulls[i])
                             for i in range(len(self.__active_arms))]
      self.__last_round = True
    else:
      # pulls assigned to each arm
      pulls = math.floor(self.budget() /
                         (len(self.__active_arms) * self.__total_rounds))
      self.__last_actions = [(arm_id, pulls)
                             for arm_id in self.__active_arms]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the ordinary bandit by executing
        `self.__last_actions`.
    """
    for (ind, (rewards, _)) in enumerate(feedback):
      if rewards is not None:
        self.__pseudo_arms[self.__last_actions[ind][0]].update(rewards)
        self.__budget_left -= len(rewards)
    if self.__last_round:
      self.__best_arm = max([(arm_id, self.__pseudo_arms[arm_id].em_mean())
                             for arm_id in self.__active_arms],
                            key=lambda x: x[1])[0]
    else:
      # remove half of the arms with the worst empirical means
      sorted_active_arms = sorted(
          list(self.__active_arms),
          key=lambda x: self.__pseudo_arms[x].em_mean(),
          reverse=True)
      self.__active_arms = set(
          sorted_active_arms[:math.ceil(len(self.__active_arms) / 2)])
    self.__round += 1

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
