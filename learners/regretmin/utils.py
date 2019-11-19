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
    return self.__horizon

  def __init__(self):
    super().__init__()
    self.__goal = 'Regret minimization'

  def _goal_init(self):
    self.__rewards = 0

  def _goal_update(self, context, action, feedback):
    self.__rewards += feedback[0]

  def _one_trial(self, seed):
    """
    Input:
      pars["horizon"]: budget
      pars["mod"]: record regret after every `mod` actions
    """
    np.random.seed(seed)
    self.__horizon = self._pars["horizon"]

    ############################################################################
    # learner initialization
    self._init(self._bandit)
    ############################################################################

    agg_regret = dict()
    for t in range(self._pars["horizon"] + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self._choice(context)
        feedback = self._bandit.feed(action)
        self._update(context, action, feedback)
      if t % self._pars["mod"]:
        agg_regret[t] = self._bandit.regret(self.__rewards)
    return dict({self.name: agg_regret})
