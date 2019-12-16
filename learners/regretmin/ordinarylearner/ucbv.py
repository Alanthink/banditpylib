import numpy as np

from .utils import OrdinaryLearner

__all__ = ['UCBV']


class UCBV(OrdinaryLearner):
  """UCB-V algorithm"""

  def __init__(self, eta=1.2):
    """eta should be greater than 1"""
    self.__eta = eta

  @property
  def name(self):
    return 'UCBV'

  def _learner_init(self):
    pass

  def _learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucbv = [arm.em_mean+
           np.sqrt(2*self.__eta*arm.em_var/arm.pulls*np.log(self._t))+
           3*self.__eta*np.log(self._t)/arm.pulls for arm in self._em_arms]

    return np.argmax(ucbv)

  def _learner_update(self, context, action, feedback):
    pass
