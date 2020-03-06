from abc import abstractmethod

from .. import RegretMinimizationLearner

__all__ = ['OrdinaryMNLLearner']


class OrdinaryMNLLearner(RegretMinimizationLearner):
  """base class for learners in the MNL bandit model"""

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
      raise Exception(
          ("%s: I don't",
           " understand the bandit environment!") % self.name)

  def _model_update(self, context, action, feedback):
    pass
