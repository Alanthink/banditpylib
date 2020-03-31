from abc import abstractmethod

import numpy as np

from banditpylib.bandits.arms import EmArm
from banditpylib.bandits import LinearBanditItf
from .. import FixConfBAILearner


def mat_norm(x, A):
  return np.sqrt(np.dot(np.dot(x, A), x))


class LinearLearner(FixConfBAILearner):
  """Base class for learners in the classic bandit model

  .. inheritance-diagram:: LinearLearner
    :parts: 1
  """

  def __init__(self, pars):
    super().__init__(pars)

  def _model_reset(self):
    """local initialization"""
    if not isinstance(self._bandit, LinearBanditItf):
      raise Exception(("%s: I don't understand",
                       " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]
    # initalize em arms repr
    for k in range(self._arm_num):
      self._em_arms[k].feature = self._bandit.features[k]

  def _model_update(self, action, feedback):
    if isinstance(action, list):
      for (i, tup) in enumerate(action):
        self._em_arms[tup[0]].update(feedback[0][i], tup[1])
    else:
      self._em_arms[action].update(feedback[0])

  @abstractmethod
  def _learner_reset(self):
    pass

  @abstractmethod
  def learner_round(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass
