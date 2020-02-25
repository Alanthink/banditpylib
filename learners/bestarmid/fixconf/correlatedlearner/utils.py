from abc import abstractmethod
from absl import logging

import numpy as np

from bandits.arms import EmArm
from learners.bestarmid.fixconf import FixConfBAILearner

__all__ = ['FixConfBAILearner', 'mat_norm']


def mat_norm(x, A):
  return np.sqrt(np.dot(np.dot(x, A), x))


class CorrelatedLearner(FixConfBAILearner):
  """base class for learners in the classic bandit model"""

  # pylint: disable=I0023, W0235
  def __init__(self, pars):
    super().__init__(pars)

  def _model_init(self):
    """local initialization"""
    if self._bandit.type != 'correlatedbandit':
      logging.fatal(("%s: I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]
    # initalize em arms repr
    for k in range(self._arm_num):
      self._em_arms[k].action = self._bandit.actions[k]

  def _model_update(self, action, feedback):
    if isinstance(action, list):
      for (i, tup) in enumerate(action):
        self._em_arms[tup[0]].update(feedback[0][i], tup[1])
    else:
      self._em_arms[action].update(feedback[0])

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def learner_run(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass
