"""
Base class for a learner with goal regret minimization.

Before each trial is run, a learner must be initialized with `init`. The first
argument `bandit` is needed since during the initializaiont, a learner may want
to ask the bandit for basic information. During each time step, `learner_choice`
is called to ask the learner for choice of the action. `update` is called by the
protocol when the reward is obtained from the bandit environment.
"""
from abc import abstractmethod

from .. import Learner

__all__ = ['RegretMinimizationLearner']


class RegretMinimizationLearner(Learner):
  """base class for regret minimization learners"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'Regret Minimization'

  @property
  def rewards_def(self):
    return self.rewards

  @property
  def regret_def(self):
    return 'regret'

  def _goal_init(self):
    self.__rewards = 0

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  def _goal_update(self, context, action, feedback):
    del context, action
    self.__rewards += feedback[0]

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  def rewards(self):
    return self.__rewards

  # pylint: disable=arguments-differ
  def init(self, bandit):
    # time starts from 1
    self._bandit = bandit
    self._t = 1
    self._goal_init()
    self._model_init()
    self._learner_init()

  @abstractmethod
  def learner_choice(self, context):
    pass

  def update(self, context, action, feedback):
    self._goal_update(context, action, feedback)
    self._model_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1
