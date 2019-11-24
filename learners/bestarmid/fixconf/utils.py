from abc import abstractmethod

import numpy as np

from learners import Learner

__all__ = ['FixConfBAILearner', 'STOP']

STOP = 'stop'


class FixConfBAILearner(Learner):
  """Base class for fixed confidence best-arm-identification learners"""

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
  def _fail_prob(self):
    return self.__fail_prob

  def __init__(self):
    super().__init__()
    self.__goal = 'Fixed Confidence Best Arm Identification'

  def _goal_init(self):
    pass

  def _goal_update(self, context, action, feedback):
    pass

  def _one_trial(self, seed):
    np.random.seed(seed)
    results = []
    for fail_prob in self._pars['fail_probs']:
      self.__fail_prob = fail_prob

      ##########################################################################
      # learner initialization
      self._init(self._bandit)
      ##########################################################################

      tot_samples = 0

      while True:
        context = self._bandit.context
        action = self._choice(context)
        if action == STOP:
          break
        feedback = self._bandit.feed(action)
        self._update(context, action, feedback)
        if isinstance(action, list):
          tot_samples += sum([tup[1] for tup in action])
        else:
          tot_samples += 1

      regret = self._bandit.best_arm_regret(self._best_arm())
      results.append(dict({self.name: [self.__fail_prob, tot_samples, regret]}))
    return results
