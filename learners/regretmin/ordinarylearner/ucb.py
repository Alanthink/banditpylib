import numpy as np

from .utils import OrdinaryLearner

__all__ = ['UCB']


class UCB(OrdinaryLearner):
  """UCB"""

  @property
  def name(self):
    return self.__name

  def __init__(self, alpha=2):
    super().__init__()
    self.__alpha = alpha
    self.__name = 'UCB'

  def _learner_init(self):
    pass

  def _choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t-1))
        for arm in self._em_arms]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass
