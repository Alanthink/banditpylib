import numpy as np

from .utils import DecentralizedOrdinaryLearner


class FUCB(DecentralizedOrdinaryLearner):
  """Friendly UCB policy"""

  def __init__(self, pars):
    super().__init__(pars)
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2
    if self.__alpha <= 0:
      raise Exception('%s: alpha should be greater than 0!' % self.name)

  @property
  def _name(self):
    return 'FUCB'

  def _learner_reset(self):
    pass

  def broadcast_message(self, context, action, feedback):
    return [action, feedback[0]]

  def learner_step(self, context, messages):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    # ucb using global conf interval
    ucb = [(messages[arm].rewards+self._em_arms[arm].rewards)/ \
        (messages[arm].pulls+self._em_arms[arm].pulls)+ \
        np.sqrt(self.__alpha / (messages[arm].pulls+self._em_arms[arm].pulls)*\
          np.log(self._t - 1)) for arm in range(self._arm_num)]
    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
