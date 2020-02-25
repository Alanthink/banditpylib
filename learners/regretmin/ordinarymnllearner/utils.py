from abc import abstractmethod
from absl import logging

from learners.regretmin import RegretMinimizationLearner

__all__ = ['OrdinaryMNLLearner']


class OrdinaryMNLLearner(RegretMinimizationLearner):
  """Base class for learners in the MNL bandit model"""

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
    if self._bandit.type != 'ordinarymnlbandit':
      logging.fatal(
          ("(%s) I don't",
           " understand the bandit environment!") % self.name)

  def _model_update(self, context, action, feedback):
    pass
