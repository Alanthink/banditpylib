import math
import numpy as np

from absl import logging

from .utils import OrdinaryLearner

__all__ = ['SR']


class SR(OrdinaryLearner):
  """successive elimination"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'SR'

  def _learner_init(self):
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

  def learner_run(self):
    if (self._budget < self._arm_num):
      # randomly output an arm
      self.__best_arm = np.random.randint(self._arm_num)
      return
    if (self._budget == self._arm_num):
      for ind in range(self._arm_num):
        feedback = self._bandit.feed(ind)
        self._model_update(ind, feedback)
      self.__best_arm = max([(ind, arm.em_mean)
        for (ind, arm) in enumerate(self._em_arms)], key=lambda x:x[1])[0]
      return

    active_arms = list(range(self._arm_num))
    budget_left = self._budget
    for k in range(1, self._arm_num-1):
      action = [(ind, self.__pulls_per_round[k]) for ind in active_arms]
      feedback = self._bandit.feed(action)
      self._model_update(action, feedback)
      budget_left -= (self.__pulls_per_round[k]*len(active_arms))
      # remove the arm with the least mean
      arm_to_remove = min([(ind, self._em_arms[ind].em_mean)
          for ind in active_arms], key=lambda x:x[1])[0]
      active_arms = [ind for ind in active_arms if (ind != arm_to_remove)]

    # check if there is only two active arms left in the last round
    if len(active_arms) != 2:
      logging.fatal('The last round should have only 2 arms!')

    action = [(active_arms[0], budget_left//2)]
    budget_left -= budget_left//2
    action.append((active_arms[1], budget_left))
    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)

    if self._em_arms[active_arms[0]].em_mean > \
        self._em_arms[active_arms[1]].em_mean:
      self.__best_arm = active_arms[0]
    else:
      self.__best_arm = active_arms[1]
    return

  def best_arm(self):
    return self.__best_arm
