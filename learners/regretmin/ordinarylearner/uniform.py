from .utils import OrdinaryLearner

__all__ = ['Uniform']


class Uniform(OrdinaryLearner):
  """sample each arm the same number of times"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'Uniform'

  def _learner_init(self):
    pass

  def learner_choice(self, context):
    """return an arm to pull"""
    return (self._t-1) % self._arm_num

  def _learner_update(self, context, action, feedback):
    pass
