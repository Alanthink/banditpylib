import math

import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['FlilUCB_heur']


class FlilUCB_heur(DecentralizedOrdinaryLearner):
  """friendly lilUCB heuristic"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def name(self):
    if self._name:
      return self._name
    return 'FlilUCB_heur'

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
    return self.__message

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
      # ucb using global conf interval
      ucb = [(messages[arm].rewards+self._em_arms[arm].rewards) / \
          (messages[arm].pulls+self._em_arms[arm].pulls) +
             self.__bonus(messages[arm].pulls+self._em_arms[arm].pulls)
             for arm in range(self._arm_num)]
      action = np.argmax(ucb)

    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)
    self.__message = [action, feedback[0]]
    self.__t += 1
    return action

  def best_arm(self):
    return max([(ind, arm.pulls)
                for (ind, arm) in enumerate(self._em_arms)],
               key=lambda x: x[1])[0]
