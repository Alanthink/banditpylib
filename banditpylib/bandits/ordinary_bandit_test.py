import google.protobuf.text_format as text_format

from banditpylib.arms import BernoulliArm
from banditpylib.data_pb2 import Actions
from banditpylib.learners import MaxReward, BestArmId
from .ordinary_bandit import OrdinaryBandit


class TestOrdinaryBandit:
  """Test ordinary bandit"""
  def test_simple_run(self):
    means = [0, 1]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = OrdinaryBandit(arms)
    ordinary_bandit.reset()
    # Pull arm 0 for 100 times
    actions = text_format.Parse(
        """
      arm_pulls_pairs {
        arm {
          id: 0
        }
        pulls: 100
      }
      """, Actions())
    ordinary_bandit.feed(actions)
    assert ordinary_bandit.regret(MaxReward()) == 100
    assert ordinary_bandit.regret(BestArmId(best_arm=1)) == 0
