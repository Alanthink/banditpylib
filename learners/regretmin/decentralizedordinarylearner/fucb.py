from absl import logging

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
      logging.fatal('%s: alpha should be greater than 0!' % self._name)

  def _learner_init(self):
    pass

  def broadcast_message(self, context, action, feedback):
    return [action, feedback[0]]

  def learner_choice(self, context, messages):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    messages = [list(m.values())[0] for m in messages]
    arm_scores = np.array(messages).transpose()
    _, pulls = np.unique(arm_scores[0], return_counts=True)
    rew = [arm_scores[1][arm_scores[0] == a].sum()
           for a in range(self._arm_num)]

    ucb = [rew[arm] / pulls[arm] +
           np.sqrt(self.__alpha / pulls[arm] * np.log(len(messages) - 1))
           for arm in range(self._arm_num)]
    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
