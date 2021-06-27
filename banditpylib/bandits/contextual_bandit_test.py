import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Actions
from banditpylib.learners import MaximizeTotalRewards
from .contextual_bandit import ContextualBandit
from .contextual_bandit_utils import RandomContextGenerator


class TestContextualBandit:
  """Test contextual bandit"""
  def test_contextual_bandit(self):
    arm_num = 10
    dimension = 10
    contextual_bandit = ContextualBandit(
        RandomContextGenerator(arm_num=arm_num, dimension=dimension))
    contextual_bandit.reset()
    for _ in range(20):
      _ = contextual_bandit.context
      contextual_bandit.feed(
          text_format.Parse(
              """
      arm_pulls {
        arm {
          id: 1
        }
        times: 1
      }
      """, Actions()))

    assert contextual_bandit.regret(MaximizeTotalRewards()) <= 20
