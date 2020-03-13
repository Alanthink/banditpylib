import numpy as np

from .utils import OrdinaryLearner

__all__ = ['MOSS']


class MOSS(OrdinaryLearner):
  """MOSS"""

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def name(self):
    if self._name:
      return self._name
    return 'MOSS'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+
           np.sqrt(
               max(0, np.log(self._horizon/(self._arm_num*arm.pulls)))
               /arm.pulls) for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
