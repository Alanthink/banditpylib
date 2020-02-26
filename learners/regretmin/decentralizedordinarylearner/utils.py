from abc import abstractmethod
from absl import logging

from bandits.arms import EmArm
from learners.regretmin import RegretMinimizationLearner

__all__ = ['DecentralizedOrdinaryLearner']


class DecentralizedOrdinaryLearner(RegretMinimizationLearner):
  """base class for learners in the classic bandit model"""

  def __init__(self, pars):
    super().__init__(pars)

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  # pylint: disable=arguments-differ
  def learner_choice(self, context, messages):
    pass

  @abstractmethod
  def broadcast_message(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  def _model_init(self):
    """local initialization"""
    if self._bandit.type != 'ordinarybandit':
      logging.fatal(("%s: I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, context, action, feedback):
    self._em_arms[action].update(feedback[0])
