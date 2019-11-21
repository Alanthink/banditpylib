from abc import abstractmethod
from absl import logging

from learners.regretmin import RegretMinimizationLearner

__all__ = ['OrdinaryMNLLearner']


class OrdinaryMNLLearner(RegretMinimizationLearner):
  """Base class for learners in the MNL bandit model"""

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def _choice(self, context):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  def __init__(self):
    super().__init__()

  def _model_init(self):
    if self._bandit.type != 'ordinarymnlbandit':
      logging.fatal(
          ("(ExplorationExploitation) I don't",
           " understand the bandit environment!"))

  def _model_update(self, context, action, feedback):
    pass
