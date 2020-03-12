from .utils import OrdinaryLearner

__all__ = ['Uniform']


class Uniform(OrdinaryLearner):
  """sample each arm the same number of times"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def name(self):
    if self._name:
      return self._name
    return 'Uniform'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    """return an arm to pull"""
    return (self._t-1) % self._arm_num

  def _learner_update(self, context, action, feedback):
    pass
