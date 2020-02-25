import math

import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['DECEN']


class DECEN(DecentralizedOrdinaryLearner):
  """Friendly lilUCB heuristic"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'DECEN'

  def _learner_init(self):
    # alg parameters suggested by the paper
    self.__beta = 0.5; self.__a = 1+10/self._arm_num; self.__eps = 0
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

    messages = [list(m.values())[0] for m in messages] # list of binary strings, each of length K

    


    return action

  def learner_run(self):
    pass

  def best_arm(self):
    return max([(ind, arm.pulls)
                for (ind, arm) in enumerate(self._em_arms)], key=lambda x: x[1])[0]
