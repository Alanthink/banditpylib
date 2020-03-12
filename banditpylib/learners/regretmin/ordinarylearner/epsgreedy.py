import numpy as np

from .utils import OrdinaryLearner

__all__ = ['EpsGreedy']


class EpsGreedy(OrdinaryLearner):
  """Epsilon-Greedy Algorithm
  With probability eps/t do uniform sampling and with the left probability,
  pull arm with the maximum empirical mean.
  """

  def __init__(self, pars):
    super().__init__(pars)
    self.__eps = float(pars['eps']) if 'eps' in pars else 1

  @property
  def name(self):
    if self._name:
      return self._name
    return 'EpsilonGreedy'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    rand = np.random.random_sample()
    if rand <= self.__eps/self._t:
      return np.random.randint(self._arm_num)
    return np.argmax(np.array([arm.em_mean for arm in self._em_arms]))

  def _learner_update(self, context, action, feedback):
    pass
