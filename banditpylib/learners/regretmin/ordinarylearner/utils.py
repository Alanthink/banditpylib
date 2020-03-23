from abc import abstractmethod

from banditpylib.bandits.arms import EmArm
from banditpylib.bandits import OrdinaryBanditItf
from .. import RegretMinimizationLearner

__all__ = ['OrdinaryLearner']


class OrdinaryLearner(RegretMinimizationLearner):
  """base class for learners in the ordinary multi-armed bandit"""

  def __init__(self, pars):
    super().__init__(pars)

  @abstractmethod
  def _learner_reset(self):
    pass

  @abstractmethod
  def learner_step(self, context):
    """return an arm to play at time ``self._t``

    Return:
      an integer in [0, ``self._arm_num``)
    """

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  def _model_reset(self):
    if not isinstance(self._bandit, OrdinaryBanditItf):
      raise Exception(("%s: I don't understand",
                       " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, context, action, feedback):
    self._em_arms[action].update(feedback[0])
