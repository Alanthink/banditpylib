from abc import abstractmethod

import numpy as np

from absl import logging

from learners import Learner

__all__ = ['FixBudgetBAILearner']


class FixBudgetBAILearner(Learner):
  """Base class for fixed budget best-arm-identification learners"""

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def _choice(self, context):
    pass

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _best_arm(self):
    pass

  @property
  def goal(self):
    return self.__goal

  @property
  def _budget(self):
    return self.__budget

  def __init__(self):
    super().__init__()
    self.__goal = 'Fixed Budget Best Arm Identification'

  def _goal_init(self):
    pass

  def _goal_update(self, context, action, feedback):
    pass

  def _one_trial(self, seed):
    np.random.seed(seed)
    results = []
    for budget in self._pars['budgets']:
      self.__budget = budget

      ##########################################################################
      # learner initialization
      self._init(self._bandit)
      ##########################################################################

      budget_remain = self.__budget

      while True:
        context = self._bandit.context
        action = self._choice(context)
        if action == 'stop':
          break
        feedback = self._bandit.feed(action)
        self._update(context, action, feedback)
        if isinstance(action, list):
          budget_remain -= sum([tup[1] for tup in action])
        else:
          budget_remain -= 1
        if budget_remain < 0:
          logging.fatal('%s uses more than the given budget!' % self.name)

      regret = self._bandit.best_arm_regret(self._best_arm())
      results.append(dict({self.name: [self.__budget, regret]}))
    return results
