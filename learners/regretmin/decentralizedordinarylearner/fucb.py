import numpy as np

from .utils import DecentralizedOrdinaryLearner

__all__ = ['FUCB']


class FUCB(DecentralizedOrdinaryLearner):
  """Friendly UCB"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'FUCB'

  def _learner_init(self):
    pass

  def _broadcast_message(self, context, action, feedback):
    return [action, feedback[0]]

  def learner_choice(self, context, messages):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    messages = [list(m.values())[0] for m in messages]
    arms = [m[0] for m in messages]
    scores = [m[1] for m in messages]

    pulls = [arms.count(a) for a in range(self._arm_num)]
    rew = [0 for _ in range(self._arm_num)]

    for i, a in enumerate(arms):
      rew[a] += scores[i]

    ucb = [rew[arm] / pulls[arm] +
            np.sqrt(self.__alpha / pulls[arm] * np.log(len(arms) - 1))
            for arm in range(self._arm_num)]
    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
