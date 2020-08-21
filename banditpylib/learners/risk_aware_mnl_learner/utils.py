from banditpylib.learners import Learner


# pylint: disable=W0223
class RiskAwareMNLLearner(Learner):
  """Base class for risk-aware learners in the ordinary mnl bandit
  """

  def set_horizon(self, horizon: int):
    self._horizon = horizon

  def regret(self, bandit) -> float:
    """
    Return:
      regret compared with the optimal policy
    """
    return bandit.regret()
