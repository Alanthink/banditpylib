from abc import abstractmethod

from absl import logging

from arms import EmArm
from bandits import Bandit
from learners.regretmin import RegretMinimizationLearner

__all__ = ['DecentralizedLearner']


class DecentralizedLearner(RegretMinimizationLearner):
  """Base class for learners in the classic bandit model"""

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def _learner_choice(self, context, messages):
    pass

  @abstractmethod
  def _broadcast_message(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  def init(self, bandit):
    if not isinstance(bandit, Bandit):
      logging.fatal('Not a legimate bandit!')

    self._bandit = bandit
    self._init = False

  def _model_init(self):
    """local initialization"""
    if self._bandit.type != 'ordinarybandit':
      logging.fatal(("(%s) I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, context, action, feedback):
    self._em_arms[action].update(feedback[0])
