import math

import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['FlilUCB_heur']


class FlilUCB_heur(DecentralizedOrdinaryLearner):
  """Friendly lilUCB heuristic"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'FlilUCB_heur'

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
    return [action, feedback[0]]

  def learner_choice(self, messages):
    """return an arm to pull"""
    if self.__t <= self._arm_num:
      self.__t += 1
      return (self.__t - 1) % self._arm_num

    messages = [list(m.values())[0] for m in messages]
    arms = [m[0] for m in messages]
    scores = [m[1] for m in messages]

    pulls = [arms.count(a) for a in range(self._arm_num)]
    rew = [0 for _ in range(self._arm_num)]

    for i, a in enumerate(arms):
      rew[a] += scores[i]

    for k, arm in enumerate(self._em_arms):
      if arm.pulls >= (1+self.__a*(self.__t-arm.pulls)):
        return -1

    # Decentralized UCB
    #ucb = [rew[arm] / pulls[arm] +
    #        self.__bonus(pulls[arm])
    #        for arm in range(self._arm_num)]

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
        for (ind, arm) in enumerate(self._em_arms)], key=lambda x:x[1])[0]