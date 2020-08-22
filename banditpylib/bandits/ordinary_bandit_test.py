from banditpylib.arms import BernoulliArm
from .ordinary_bandit import OrdinaryBandit


class TestOrdinaryBandit:
  """Test ordinary bandit"""

  def test_simple_run(self):
    means = [0, 1]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = OrdinaryBandit(arms)
    ordinary_bandit.reset()
    # pull arm 0 for 100 times
    ordinary_bandit.feed([(0, 100)])
    assert ordinary_bandit.regret() == 100
    assert ordinary_bandit.best_arm_regret(1) == 0
