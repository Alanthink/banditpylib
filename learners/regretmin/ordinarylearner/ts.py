import numpy as np

from .utils import OrdinaryLearner

__all__ = ['TS']


class TS(OrdinaryLearner):
  """Thompson Sampling"""

  def __init__(self):
    pass

  @property
  def name(self):
    return 'Thompson Sampling'

  def _learner_init(self):
    pass

  def learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    # each arm has a uniform prior B(1, 1)
    vir_means = np.zeros(self._arm_num)
    for arm in range(self._arm_num):
      a = 1 + self._em_arms[arm].rewards
      b = 1 + self._em_arms[arm].pulls - self._em_arms[arm].rewards
      vir_means[arm] = np.random.beta(a, b)

    return np.argmax(vir_means)

  def _learner_update(self, context, action, feedback):
    pass
