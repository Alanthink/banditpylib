"""
Learners under the classic bandit model.
"""

from abc import abstractmethod

import math
import numpy as np

from absl import logging

from arms import EmArm
from .utils import FixBudgetBAILearner

__all__ = ['Uniform', 'SR']


class OrdinaryLearner(FixBudgetBAILearner):
  """Base class for learners in the classic bandit model"""

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def choice(self, context):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _best_arm(self):
    pass

  def __init__(self):
    super().__init__()

  def _model_init(self):
    """local initialization"""
    if self._bandit.type != 'ordinarybandit':
      logging.fatal(("(%s) I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, context, action, feedback):
    for i in range(action):
      self._em_arms[action[i][0]].update(feedback[i], action[i][1])


class Uniform(OrdinaryLearner):
  """Naive uniform algorithm: sample each arm the same number of times"""

  @property
  def name(self):
    return self.__name

  def __init__(self):
    super().__init__()
    self.__name = 'Uniform'

  def _learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    return (self._t-1) % self._arm_num

  def _learner_update(self, context, action, feedback):
    pass

  def _best_arm(self):
    return np.argmax([arm.em_mean for arm in self._em_arms])


class SR(OrdinaryLearner):
  """Successive Elimination"""

  @property
  def name(self):
    return self.__name

  def __init__(self):
    super().__init__()
    self.__name = 'SR'

  def _learner_init(self):
    self.__simple_mode = (self.budget <= self._arm_num)
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
            1/self.__bar_log_K*(self.budget-self._arm_num)/(self._arm_num+1-k)))
        if k == 1:
          self.__pulls_per_round.append(nk[-1])
        else:
          self.__pulls_per_round.append(nk[-1]-nk[-2])

      self.__budget_left = self.budget
      self.__k = 0
      self.__active_arms = list(range(self._arm_num))

  def choice(self, context):
    """return an arm to pull"""
    if self.__budget_left <= 0:
      return 'stop'

    if self.__simple_mode:
      if (self.budget < self._arm_num):
        self.__best_arm = np.random.randint(self._arm_num)
        return 'stop'

      self.__budget_left = 0
      return [(arm, 1) for arm in range(self._arm_num)]

    # normal mode (run in rounds)
    self.__k += 1
    if self.__k != (self._arm_num-1):
      self.__budget_left -= (self.__budget_left[self.__k]*len(self.__active_arms))
      return [(arm, self.__pulls_per_round[self.__k])
              for arm in range(self._arm_num)]

    # check there is only two active arms left
    if len(self.__active_arms) != 2:
      logging.fatal('The last round should have only 2 arms!')

    action = []
    action.append((self.__active_arms[0], self.__budget_left//2))
    self.__budget_left -= self.__budget_left//2
    action.append((self.__active_arms[1], self.__budget_left//2))
    self.__budget_left = 0
    return action

  def _learner_update(self, context, action, feedback):
    if self.__simple_mode:
      self.__best_arm = np.argmax(
          np.array([self._em_arms[arm].em_mean for arm in self.__active_arms]))
    else:
      # remove the arm with the least mean
      em_means = [(arm, self._em_arms[arm].em_mean)
          for arm in self.__active_arms]
      arm_to_remove = max(em_means, key=lambda x:x[1])[0]
      arms = [arm for arm in self.__active_arms if arm != arm_to_remove]
      if len(arms) == 1:
        self.__best_arm = arms[0]

  def _best_arm(self):
    return self.__best_arm
