from banditpylib.learners import Learner


# pylint: disable=W0223
class OrdinaryLearner(Learner):
  """Base class for learners in the ordinary multi-armed bandit
  """

  def regret(self, bandit) -> float:
    """
    Return:
      regret compared with the optimal policy
    """
    return bandit.regret()
