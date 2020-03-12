import math

import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['SlilUCB_heur']


class SlilUCB_heur(DecentralizedOrdinaryLearner):
  """selfish lilUCB heuristic"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def name(self):
    if self._name:
      return self._name
    return 'SlilLUCB_heur'

  def __bonus(self, times):
    if (1+self.__eps)*times == 1:
      return math.inf
    return (1+self.__beta)*(1+math.sqrt(self.__eps))* \
        math.sqrt(2*(1+self.__eps)* \
        math.log(math.log((1+self.__eps)*times)/self.__delta)/times)

  def _learner_reset(self):
    # alg parameters suggested by the paper
    self.__beta = 0.5
    self.__a = 1+10/self._arm_num
    self.__eps = 0
    self.__delta = self._fail_prob/5
    # total number of pulls used
    self.__t = 0

  def broadcast_message(self):
    return None

  def learner_round(self, messages):
    """return an arm to pull"""
    # to avoid naive stop
    if self.__t > self._arm_num:
      for arm in self._em_arms:
        if arm.pulls >= (1+self.__a*(self.__t-arm.pulls)):
          return -1
    if self.__t < self._arm_num:
      action = self.__t % self._arm_num
    else:
      ucb = np.array([arm.em_mean+self.__bonus(arm.pulls)
                      for arm in self._em_arms])
      action = np.argmax(ucb)
    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)
    self.__t += 1
    return action

  def best_arm(self):
    return max([(ind, arm.pulls)
                for (ind, arm) in enumerate(self._em_arms)],
               key=lambda x: x[1])[0]
