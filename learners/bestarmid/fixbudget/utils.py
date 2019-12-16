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

  @property
  def goal(self):
    return 'Fixed Budget Best Arm Identification'

  @property
  def _budget(self):
    return self.__budget

  def _goal_init(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def _learner_run(self):
    pass

  @abstractmethod
  def _best_arm(self):
    pass

  def __init(self):
    self._bandit.init()
    self._goal_init()
    self._model_init()
    self._learner_init()

  def _one_trial(self, seed):
    np.random.seed(seed)
    results = []
    for budget in self._pars['budgets']:
      self.__budget = budget

      ##########################################################################
      # initialization
      self.__init()
      ##########################################################################

      self._learner_run()
      if self._bandit.tot_samples > budget:
        logging.fatal('%s uses more than the given budget!' % self.name)

      regret = self._bandit.best_arm_regret(self._best_arm())
      results.append(dict({self.name: [budget, regret]}))
    return results
