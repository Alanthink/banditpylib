import math

import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['SlilUCB_heur']


class SlilUCB_heur(DecentralizedOrdinaryLearner):
  """Selfish lilUCB heuristic"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'SlilLUCB_heur'

  def __bonus(self, times):
    if (1+self.__eps)*times == 1:
      return math.inf
    return (1+self.__beta)*(1+math.sqrt(self.__eps))* \
        math.sqrt(2*(1+self.__eps)* \
        math.log(math.log((1+self.__eps)*times)/self.__delta)/times)

  def _learner_init(self):
    # alg parameters suggested by the paper
    self.__beta = 0.5; self.__a = 1+10/self._arm_num; self.__eps = 0
    self.__delta = self._fail_prob/5
    # total number of pulls used
    self.__t = 0

  def _broadcast_message(self, action, feedback):
    return None

  def learner_choice(self, messages):
    """return an arm to pull"""
    if self.__t <= self._arm_num:
      self.__t += 1
      return (self.__t - 1) % self._arm_num
    for arm in self._em_arms:
      if arm.pulls >= (1+self.__a*(self.__t-arm.pulls)):
        return -1

    ucb = np.array([arm.em_mean+self.__bonus(arm.pulls) \
        for arm in self._em_arms])
    self.__t += 1
    action = np.argmax(ucb)
    return action

  def learner_run(self):
    pass

  def best_arm(self):
    return max([(ind, arm.pulls)
        for (ind, arm) in enumerate(self._em_arms)], key=lambda x:x[1])[0]