from abc import abstractmethod

from .. import Learner


class BAILearner(Learner):
  """
  Base class for a learner with goal best arm identification.

  Instead of ``learner_step``, pure exploration learners implement
  ``learner_round`` method which essentially means a round. The learner
  has to decide by him/herself when to stop.
  """

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
  def learner_round(self):
    """Run one round"""

  @abstractmethod
  def reset(self, bandit, stop_cond):
    pass
