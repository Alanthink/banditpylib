"""
Base class for a learner with goal best arm identification.

Instead of `learner_step`, pure exploration learners implement `learner_round`
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
  def goal(self):
    return 'Best Arm Identification'

  @abstractmethod
  def best_arm(self):
    pass

  @abstractmethod
  def learner_round(self):
    """one round of running"""

  @abstractmethod
  def reset(self):
    pass
