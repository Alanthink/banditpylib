from abc import abstractmethod

from .. import BAILearner

__all__ = ['FixConfBAILearner']


class FixConfBAILearner(BAILearner):
  """base class for fixed confidence best arm identification learners"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'Fix Confidence Best Arm Identification'

  def _goal_reset(self):
    pass

  @abstractmethod
  def _model_reset(self):
    pass

  @abstractmethod
  def _learner_reset(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass

  def reset(self, bandit, stop_cond):
    self._bandit = bandit
    self._fail_prob = stop_cond
    self._goal_reset()
    self._model_reset()
    self._learner_reset()
