from abc import abstractmethod

import numpy as np

from learners import Learner

__all__ = ['FixConfBAILearner']


class FixConfBAILearner(Learner):
  """Base class for fixed confidence best-arm-identification learners"""

  @property
  @abstractmethod
  def name(self):
    pass

  @property
  def goal(self):
    return 'Fixed Confidence Best Arm Identification'

  @property
  def _fail_prob(self):
    return self.__fail_prob

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
    for fail_prob in self._pars['fail_probs']:
      self.__fail_prob = fail_prob

      ##########################################################################
      # initialization
      self.__init()
      ##########################################################################

      self._learner_run()

      regret = self._bandit.best_arm_regret(self._best_arm())
      results.append(
          dict({self.name: [fail_prob, self._bandit.tot_samples, regret]}))
    return results
