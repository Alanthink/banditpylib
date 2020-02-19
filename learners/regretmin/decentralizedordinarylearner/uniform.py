from .utils import DecentralizedOrdinaryLearner

__all__ = ['Uniform']


class Uniform(DecentralizedOrdinaryLearner):
  """Naive uniform algorithm: sample each arm the same number of times"""

  def __init__(self):
    pass

  @property
  def name(self):
    return 'Uniform'

  def _learner_init(self):
    pass

  def broadcast_message(self, context, action, feedback):
    return None

  def learner_choice(self, context, messages):
    """return an arm to pull"""
    return (self._t - 1) % self._arm_num

  def _learner_update(self, context, action, feedback):
    pass
