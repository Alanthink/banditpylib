import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Feedback
from .explore_then_commit import ExploreThenCommit


class TestEXP3:
  """Test EXP3 policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 10
    learner = ExploreThenCommit(arm_num=arm_num, T_prime=6)
    learner.reset()

    for _ in range(1, horizon + 1):
      actions = learner.actions()
      assert len(actions.arm_pulls_pairs) == 1
      arm_pulls_pair = actions.arm_pulls_pairs[0]
      arm_id = arm_pulls_pair.arm.id
      assert arm_pulls_pair.pulls == 1
      learner.update(
          text_format.Parse(
              """
        arm_rewards_pairs <
          arm <
            id: {arm_id}
          >
          rewards: 0
        >
        """.format(arm_id=arm_id), Feedback()))
