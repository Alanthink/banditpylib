import math

import numpy as np

from .utils import OrdinaryLearner

__all__ = ['MOSS']


class MOSS(OrdinaryLearner):
  r"""MOSS policy :cite:`audibert2009minimax`.

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in [0, N-1]} \left\{ \hat{\mu}_i(t) + \sqrt{
    \frac{\mathrm{max}(\ln( \frac{T}{N T_i(t)} ), 0 ) }{T_i(t)} } \right\}

  .. note::
    MOSS will is using time horizon ``self._horizon``. Reward has to be bounded
    in [0, 1].
  """

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def _name(self):
    return 'MOSS'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+
           math.sqrt(
               max(0, math.log(self._horizon/(self._arm_num*arm.pulls)))
               /arm.pulls) for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
