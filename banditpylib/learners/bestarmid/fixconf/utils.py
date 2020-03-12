from abc import abstractmethod

from .. import BAILearner

__all__ = ['FixConfBAILearner']


class FixConfBAILearner(BAILearner):
  """base class for fixed confidence best arm identification learners"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'FixConfBAI'

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

  # pylint: disable=arguments-differ
  def reset(self, bandit, fail_prob):
    self._bandit = bandit
    self._fail_prob = fail_prob
    self._goal_reset()
    self._model_reset()
    self._learner_reset()
