import math
import numpy as np

from absl import logging

from learners.bestarmid.fixbudget import STOP
from .utils import OrdinaryLearner

__all__ = ['SR']


class SR(OrdinaryLearner):
  """Successive Elimination"""

  @property
  def name(self):
    return self.__name

  def __init__(self):
    super().__init__()
    self.__name = 'SR'

  def _learner_init(self):
    self.__simple_mode = (self._budget <= self._arm_num)
    if not self.__simple_mode:
      # calculate bar_log_K
      self.__bar_log_K = 0.5
      for i in range(2, self._arm_num+1):
        self.__bar_log_K += (1/i)

      # calculate pulls_per_round
      self.__pulls_per_round = [-1]
      nk = [-1]
      for k in range(1, self._arm_num):
        nk.append(math.ceil(
            1/self.__bar_log_K*(self._budget-self._arm_num)/(self._arm_num+1-k)))
        if k == 1:
          self.__pulls_per_round.append(nk[-1])
        else:
          self.__pulls_per_round.append(nk[-1]-nk[-2])

      # __k is the round index
      self.__k = 0
      self.__active_arms = list(range(self._arm_num))

    self.__budget_left = self._budget

  def _choice(self, context):
    """return an arm to pull"""
    if self.__budget_left <= 0:
      return STOP

    if self.__simple_mode:
      if (self._budget < self._arm_num):
        self.__best_arm = np.random.randint(self._arm_num)
        return STOP

      self.__budget_left = 0
      return [(arm, 1) for arm in range(self._arm_num)]

    # normal mode (run in rounds)
    self.__k += 1
    if self.__k != (self._arm_num-1):
      self.__budget_left -= (
          self.__pulls_per_round[self.__k]*len(self.__active_arms))
      return [(ind, self.__pulls_per_round[self.__k])
              for ind in self.__active_arms]

    # check if there is only two active arms left in the last round
    if len(self.__active_arms) != 2:
      logging.fatal('The last round should have only 2 arms!')

    action = []
    action.append((self.__active_arms[0], self.__budget_left//2))
    self.__budget_left -= self.__budget_left//2
    action.append((self.__active_arms[1], self.__budget_left))
    self.__budget_left = 0
    return action

  def _learner_update(self, context, action, feedback):
    if self.__simple_mode:
      self.__best_arm = np.argmax(
          np.array([self._em_arms[arm].em_mean for arm in self.__active_arms]))
    else:
      # remove the arm with the least mean
      arm_to_remove = min([(ind, self._em_arms[ind].em_mean)
          for ind in self.__active_arms], key=lambda x:x[1])[0]
      self.__active_arms = [
          ind for ind in self.__active_arms if (ind != arm_to_remove)]
      if len(self.__active_arms) == 1:
        self.__best_arm = self.__active_arms[0]

  def _best_arm(self):
    return self.__best_arm
