from absl import logging

import numpy as np

from .utils import OrdinaryLearner

__all__ = ['UCB']


class UCB(OrdinaryLearner):
  """UCB"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'UCB'
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2
    if self.__alpha <= 0:
      logging.fatal('%s: alpha should be greater than 0!' % self._name)

  def _learner_init(self):
    pass

  def learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t-1))
           for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
