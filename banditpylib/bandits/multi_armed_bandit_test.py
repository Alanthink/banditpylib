import google.protobuf.text_format as text_format

from banditpylib.arms import BernoulliArm
from banditpylib.data_pb2 import Arm, Actions
from banditpylib.learners import MaximizeTotalRewards, IdentifyBestArm
from .multi_armed_bandit import MultiArmedBandit


class TestOrdinaryBandit:
  """Test ordinary bandit"""
  def test_simple_run(self):
    means = [0, 1]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = MultiArmedBandit(arms)
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
    assert ordinary_bandit.regret(MaximizeTotalRewards()) == 100
    arm = Arm()
    arm.id = 1
    assert ordinary_bandit.regret(IdentifyBestArm(best_arm=arm)) == 0
