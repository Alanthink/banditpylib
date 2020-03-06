import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['FUCB']


class FUCB(DecentralizedOrdinaryLearner):
  """friendly UCB"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'FUCB'
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2
    if self.__alpha <= 0:
      raise Exception('%s: alpha should be greater than 0!' % self._name)

  def _learner_init(self):
    pass

  def broadcast_message(self, context, action, feedback):
    return [action, feedback[0]]

  def learner_choice(self, context, messages):
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
