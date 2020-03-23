from . import OrdinaryLearner

__all__ = ['Uniform']


class Uniform(OrdinaryLearner):
  """Uniform Sampling policy.

  Sample each arm the same number of times.
  """

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def _name(self):
    return 'Uniform'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    return (self._t-1) % self._arm_num

  def _learner_update(self, context, action, feedback):
    pass
