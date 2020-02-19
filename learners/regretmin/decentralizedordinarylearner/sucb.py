import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['SUCB']


class SUCB(DecentralizedOrdinaryLearner):
  """Selfish UCB"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'SUCB'

  def _learner_init(self):
    pass

  def broadcast_message(self, context, action, feedback):
    return None

  def learner_choice(self, context, messages):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t - 1) % self._arm_num

    ucb = [arm.em_mean + np.sqrt(self.__alpha / arm.pulls * np.log(self._t - 1))
            for arm in self._em_arms]
    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
