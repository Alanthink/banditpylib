from absl import logging

import numpy as np

from .utils import OrdinaryLearner

__all__ = ['UCBV']


class UCBV(OrdinaryLearner):
  """UCB-V algorithm"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'UCBV'
    self.__eta = float(pars['eta']) if 'eta' in pars else 1.2
    if self.__eta <= 1:
      logging.fatal('%s: eta should be greater than 1!' % self._name)

  def _learner_init(self):
    pass

  def learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucbv = [arm.em_mean+
            np.sqrt(2*self.__eta*arm.em_var/arm.pulls*np.log(self._t))+
            3*self.__eta*np.log(self._t)/arm.pulls for arm in self._em_arms]

    return np.argmax(ucbv)

  def _learner_update(self, context, action, feedback):
    pass
