"""
Base class for a learner with goal best arm identification.

Instead of `learner_choice`, pure exploration learners implement `learner_run`
method which essentially means a round. The learner has to decide by him/herself
when to stop.
"""
from abc import abstractmethod

from .. import Learner

__all__ = ['BAILearner']


class BAILearner(Learner):
  """base class for best arm identification learners"""

  protocol = 'SinglePlayerPEProtocol'

  def __init__(self, pars):
    super().__init__(pars)

  @property
  @abstractmethod
  def goal(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass

  @abstractmethod
  def learner_run(self):
    """one round of running"""

  @abstractmethod
  def init(self):
    pass

  @property
  def rewards_def(self):
    return self.best_arm

  @property
  def regret_def(self):
    return 'best_arm_regret'
