from abc import abstractmethod

from .. import BAILearner


class FixBudgetBAILearner(BAILearner):
  """base class for fixed budget best arm identification learners"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'Fix Budget Best Arm Identification'

  def _goal_reset(self):
    pass

  @abstractmethod
  def _model_reset(self):
    pass

  @abstractmethod
  def _learner_reset(self):
    pass

  @abstractmethod
  def learner_round(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass

  def reset(self, bandit, stop_cond):
    self._bandit = bandit
    self._budget = stop_cond
    self._goal_reset()
    self._model_reset()
    self._learner_reset()
