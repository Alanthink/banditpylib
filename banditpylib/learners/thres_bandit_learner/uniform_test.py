import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Feedback
from .uniform import Uniform


class TestUniform:
  """Test Uniform Sampling algorithm"""
  def test_actions(self):
    # Test actions are in the right range
    arm_num = 10
    budget = 15
    apt = Uniform(arm_num=arm_num, theta=0.5, eps=0)
    apt.reset()
    for _ in range(budget):
      actions = apt.actions()
      assert len(actions.arm_pulls_pairs) == 1

      arm_id = actions.arm_pulls_pairs[0].arm.id
      assert 0 <= arm_id < arm_num

      apt.update(
          text_format.Parse(
              """
        arm_rewards_pairs <
          arm <
            id: {arm_id}
          >
          rewards: 0
        >
        """.format(arm_id=arm_id), Feedback()))
