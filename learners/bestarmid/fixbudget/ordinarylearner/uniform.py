from learners.bestarmid.fixbudget import STOP
from .utils import OrdinaryLearner

__all__ = ['Uniform']


class Uniform(OrdinaryLearner):
  """Naive uniform algorithm: sample each arm the same number of times"""

  @property
  def name(self):
    return self.__name

  def __init__(self):
    super().__init__()
    self.__name = 'Uniform'

  def _learner_init(self):
    pass

  def _choice(self, context):
    """return an arm to pull"""
    if self._t <= self._budget:
      return (self._t-1) % self._arm_num
    return STOP

  def _learner_update(self, context, action, feedback):
    pass

  def _best_arm(self):
    return max([(ind, arm.em_mean)
        for (ind, arm) in enumerate(self._em_arms)], key=lambda x:x[1])[0]
