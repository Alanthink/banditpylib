import numpy as np

from .utils import OrdinaryLearner

__all__ = ['UCB']


class UCB(OrdinaryLearner):
  """UCB"""

  def __init__(self, pars):
    super().__init__(pars)
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2
    if self.__alpha <= 0:
      raise Exception('%s: alpha should be greater than 0!' % self.name)

  @property
  def _name(self):
    return 'UCB'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t-1))
           for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
