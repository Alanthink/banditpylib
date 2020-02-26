import math

from absl import logging

import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['FlilUCB_heur']


class FlilUCB_heur(DecentralizedOrdinaryLearner):
  """friendly lilUCB heuristic"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'FlilUCB_heur'
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2
    if self.__alpha <= 0:
      logging.fatal('%s: alpha should be greater than 0!' % self._name)

  def __bonus(self, times):
    if (1+self.__eps)*times == 1:
      return math.inf
    return (1+self.__beta)*(1+math.sqrt(self.__eps)) * \
            math.sqrt(2*(1+self.__eps) *
                      math.log(math.log((1+self.__eps)*times)/self.__delta)
                      /times)

  def _learner_init(self):
    # alg parameters suggested by the paper
    self.__beta = 0.5
    self.__a = 1+10/self._arm_num
    self.__eps = 0
    self.__delta = self._fail_prob/5
    # total number of pulls used
    self.__t = 0

  def broadcast_message(self, action, feedback):
    return [action, feedback[0]]

  def learner_choice(self, messages):
    """return an arm to pull"""
    if self.__t <= self._arm_num:
      self.__t += 1
      return (self.__t - 1) % self._arm_num

    messages = [list(m.values())[0] for m in messages]
    arm_scores = np.array(messages).transpose()
    _, pulls = np.unique(arm_scores[0], return_counts=True)
    rew = [arm_scores[1][arm_scores[0] == a].sum()
           for a in range(self._arm_num)]

    for _, arm in enumerate(self._em_arms):
      if arm.pulls >= (1+self.__a*(self.__t-arm.pulls)):
        return -1

    # UCB using local conf interval
    ucb = [rew[arm] / pulls[arm] +
           self.__bonus(self._em_arms[arm].pulls)
           for arm in range(self._arm_num)]

    self.__t += 1
    action = np.argmax(ucb)
    return action

  def learner_run(self):
    pass

  def best_arm(self):
    return max([(ind, arm.pulls)
                for (ind, arm) in enumerate(self._em_arms)],
               key=lambda x: x[1])[0]
