from abc import abstractmethod

from banditpylib.bandits.arms import EmArm
from banditpylib.bandits import LinearBanditItf
from .. import RegretMinimizationLearner


class CorrelatedLearner(RegretMinimizationLearner):
  """base class for learners in the classic bandit model"""

  def __init__(self, pars):
    super().__init__(pars)

  @abstractmethod
  def _learner_reset(self):
    pass

  @abstractmethod
  def learner_step(self, context):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

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

  def _model_update(self, context, action, feedback):
    self._em_arms[action].update(feedback[0])
