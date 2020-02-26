from abc import abstractmethod

from learners import Learner

__all__ = ['FixBudgetBAILearner']


class FixBudgetBAILearner(Learner):
  """base class for fixed budget best arm identification learners"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'FixedBudgetBAI'

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

  def init(self, bandit, budget):
    self._bandit = bandit
    self._budget = budget
    self._goal_init()
    self._model_init()
    self._learner_init()
