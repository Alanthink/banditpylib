from abc import abstractmethod

from ...learners import Learner

__all__ = ['BAILearner']


class BAILearner(Learner):
  """base class for best arm identification learners"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  @abstractmethod
  def goal(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass

  @property
  def reward(self):
    return self.best_arm

  @property
  def regret_def(self):
    return 'best_arm_regret'
