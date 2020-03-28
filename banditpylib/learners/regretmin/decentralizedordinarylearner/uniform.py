from .utils import DecentralizedOrdinaryLearner

__all__ = ['Uniform']


class Uniform(DecentralizedOrdinaryLearner):
  """Uniform sampling policy.

  Sample each arm in a round-robin fashion.
  """

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def _name(self):
    return 'Uniform'

  def _learner_reset(self):
    pass

  def broadcast_message(self, context, action, feedback):
    return None

  def learner_step(self, context, messages):
    """return an arm to pull"""
    return (self._t - 1) % self._arm_num

  def _learner_update(self, context, action, feedback):
    pass
