import random

import numpy as np

from .utils import OrdinaryLearner


class EpsGreedy(OrdinaryLearner):
  r"""Epsilon-Greedy policy.

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  left probability play the arm with the maximum empirical mean.
  """

  def __init__(self, pars):
    """
    Args:
      pars (dict):
        ``'eps'`` (float, *optional*): default value is 1.0. It should be
        greater than 0.
    """
    super().__init__(pars)
    self.__eps = float(pars['eps']) if 'eps' in pars else 1.0
    if self.__eps <= 0:
      raise Exception('%s: eps is less than or equal to zero!' % self.name)

  @property
  def _name(self):
    return 'EpsilonGreedy'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    rand = random.random()
    if rand <= self.__eps/self._t:
      return random.randint(0, self._arm_num-1)
    return np.argmax(np.array([arm.em_mean for arm in self._em_arms]))

  def _learner_update(self, context, action, feedback):
    pass
