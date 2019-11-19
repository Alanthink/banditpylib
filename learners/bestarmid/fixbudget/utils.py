from abc import abstractmethod

import numpy as np

from absl import logging

from learners.learner import Learner

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
  def choice(self, context):
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
  def budget(self):
    return self.budget

  def __init__(self):
    super().__init__()
    self.__goal = 'Fixed Budget Best Arm Identification'

  def _goal_init(self):
    pass

  def _goal_update(self, context, action, feedback):
    pass

  def one_trial(self, seed):
    """
    Input:
      pars["budget"]
    """
    np.random.seed(seed)
    self.__budget = self._pars["budget"]

    ############################################################################
    # learner initialization
    self.init(self._bandit)
    ############################################################################

    budget_remain = self.__budget

    while True:
      context = self._bandit.context
      action = self.choice(context)
      if action == 'stop':
        break
      feedback = self._bandit.feed(action)
      self.update(context, action, feedback)
      budget_remain -= sum(action[1])
      if budget_remain < 0:
        logging.fatal('%s uses more than the given budget!' % self.name)

    regret = self._bandit.best_arm_regret(self._best_arm())

    return dict({self.name: [self._pars["budget"], regret] })
