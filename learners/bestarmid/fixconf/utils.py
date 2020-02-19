from abc import abstractmethod

from learners import Learner

__all__ = ['FixConfBAILearner']


class FixConfBAILearner(Learner):
  """Base Class for Fixed Confidence Best Arm Identification Learners"""

  @property
  @abstractmethod
  def name(self):
    pass

  @property
  def goal(self):
    return 'FixedConfidenceBAI'

  def _goal_init(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def learner_run(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass

  def init(self, bandit, fail_prob):
    self._bandit = bandit
    self._fail_prob = fail_prob
    self._goal_init()
    self._model_init()
    self._learner_init()
