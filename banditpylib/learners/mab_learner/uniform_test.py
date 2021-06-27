import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Context, Actions, Feedback
from .uniform import Uniform


class TestUniform:
  """Test uniform policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 10
    learner = Uniform(arm_num=arm_num)
    learner.reset()

    for time in range(1, horizon + 1):
      assert learner.actions(
          Context()).SerializeToString() == text_format.Parse(
              """
        arm_pulls_pairs <
          arm <
            id: {arm_id}
          >
          pulls: 1
        >
        """.format(arm_id=(time - 1) % arm_num),
              Actions()).SerializeToString()
      learner.update(
          text_format.Parse(
              """
        arm_feedbacks <
          arm <
            id: 0
          >
          rewards: 0
        >
        """, Feedback()))
