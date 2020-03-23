import math

import numpy as np

from .utils import OrdinaryLearner

__all__ = ['UCBV']


class UCBV(OrdinaryLearner):
  r"""UCB1 policy :cite:`audibert2009exploration`.

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in [0, N-1]} \left\{ \hat{\mu}_i(t) + \sqrt{ \frac{ 2
    \hat{V}_i(t) \ln(t) }{T_i(t)} }+ \frac{ b \ln(t) }{T_i(t)} \right\}

  .. note::
    Reward has to be bounded in [0, :math:`b`].
  """

  def __init__(self, pars):
    """
    Args:
      ``pars``: a dictionary. Key ``'b'`` (*optional*) is the upper bound of
       reward. Default value is 1.0.
    """
    super().__init__(pars)
    self.__b = float(pars['b']) if 'b' in pars else 1.0
    if self.__b <= 0:
      raise Exception('%s: b should be greater than 0!' % self.name)

  @property
  def _name(self):
    return 'UCBV'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucbv = [arm.em_mean+
            math.sqrt(2*arm.em_var*math.log(self._t)/arm.pulls)+
            self.__b*math.log(self._t)/arm.pulls for arm in self._em_arms]

    return np.argmax(ucbv)

  def _learner_update(self, context, action, feedback):
    pass
