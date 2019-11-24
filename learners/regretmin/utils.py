from abc import abstractmethod

import numpy as np

from learners import Learner

__all__ = ['RegretMinimizationLearner']


class RegretMinimizationLearner(Learner):
  """Base class for regret minimization learners"""

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

  @property
  def goal(self):
    return self.__goal

  @property
  def _horizon(self):
    return self._pars['horizon']

  @property
  def __frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  def __init__(self):
    super().__init__()
    self.__goal = 'Regret minimization'

  def _goal_init(self):
    self.__rewards = 0

  def _goal_update(self, context, action, feedback):
    self.__rewards += feedback[0]

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # learner initialization
    self._init(self._bandit)
    ############################################################################

    agg_regret = dict()
    for t in range(self._horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self._choice(context)
        feedback = self._bandit.feed(action)
        self._update(context, action, feedback)
      if t % self.__frequency == 0:
        agg_regret[t] = self._bandit.regret(self.__rewards)
    return dict({self.name: agg_regret})
