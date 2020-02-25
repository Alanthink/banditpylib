from abc import abstractmethod

from absl import logging

from bandits.arms import EmArm
from learners.regretmin import RegretMinimizationLearner

__all__ = ['OrdinaryLearner']


class OrdinaryLearner(RegretMinimizationLearner):
  """Base class for learners in the classic bandit model"""

  # pylint: disable=I0023, W0235
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
    if self._bandit.type not in ['ordinarybandit', 'correlatedbandit']:
      logging.fatal(("%s: I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, context, action, feedback):
    self._em_arms[action].update(feedback[0])
