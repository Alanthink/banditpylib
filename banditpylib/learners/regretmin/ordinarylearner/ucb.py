import math

import numpy as np

from .utils import OrdinaryLearner


class UCB(OrdinaryLearner):
  r"""UCB1 policy :cite:`auer2002finite`.

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in [0, N-1]} \left\{ \hat{\mu}_i(t) + \sqrt{ \frac{
    \alpha  \ln(t) }{T_i(t)} } \right\}

  .. note::
    Reward has to be bounded in [0, 1].
  """

  def __init__(self, pars):
    """
    Args:
      ``pars`` (dict): Key ``'alpha'`` (*optional*) should be
       greater than 0. Default value is 2.0.
    """
    super().__init__(pars)
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2.0
    if self.__alpha <= 0:
      raise Exception('%s: alpha should be greater than 0!' % self.name)

  @property
  def _name(self):
    return 'UCB'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+math.sqrt(self.__alpha*math.log(self._t)/arm.pulls)
           for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
