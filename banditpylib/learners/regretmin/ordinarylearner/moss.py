import numpy as np

from .utils import OrdinaryLearner

__all__ = ['MOSS']


class MOSS(OrdinaryLearner):
  """MOSS"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'MOSS'
    if 'horizon' not in pars:
      raise Exception('%s: I need to know the horizon!' % self._name)
    else:
      self.__horizon = pars['horizon']

  def _learner_init(self):
    pass

  def learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+
           np.sqrt(
               max(0, np.log(self.__horizon/(self._arm_num*arm.pulls)))
               /arm.pulls) for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
