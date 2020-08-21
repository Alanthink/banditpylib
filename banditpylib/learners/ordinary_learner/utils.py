from abc import abstractmethod

from banditpylib.learners import Learner


class OrdinaryLearner(Learner):
  """Base class for learners in the ordinary multi-armed bandit
  """

  @abstractmethod
  def _total_rewards(self) -> float:
    """
    Return:
      total rewards obtained so far
    """

  def regret(self, bandit) -> float:
    """
    Return:
      regret compared with the optimal policy
    """
    return bandit.regret(self._total_rewards())
