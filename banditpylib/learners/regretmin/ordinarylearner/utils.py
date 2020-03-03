from abc import abstractmethod

from absl import logging

from ....bandits.arms import EmArm
from ....bandits import OrdinaryBanditItf
from ...regretmin import RegretMinimizationLearner

__all__ = ['OrdinaryLearner']


class OrdinaryLearner(RegretMinimizationLearner):
  """base class for learners in the classic bandit model"""

  def __init__(self, pars):
    super().__init__(pars)

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def learner_choice(self, context):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  def _model_init(self):
    """local initialization"""
    if not isinstance(self._bandit, OrdinaryBanditItf):
      logging.fatal(("%s: I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, context, action, feedback):
    self._em_arms[action].update(feedback[0])
