import google.protobuf.text_format as text_format

from banditpylib.arms import BernoulliArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .eps_greedy import EpsGreedy


class TestEpsGreedy:
  """Test epsilon greedy policy"""
  def test_simple_run(self):
    means = [0, 0.5, 0.7, 1]
    arms = [BernoulliArm(mean) for mean in means]
    learner = EpsGreedy(arm_num=len(arms))
    learner.reset()

    # Pull each arm once during the initial steps
    for time in range(1, len(arms) + 1):
      assert learner.actions(
          Context()).SerializeToString() == text_format.Parse(
              """
        arm_pulls <
          arm <
            id: {arm_id}
          >
          times: 1
        >
        """.format(arm_id=time - 1), Actions()).SerializeToString()
      learner.update(
          text_format.Parse(
              """
        arm_feedbacks <
          arm <
            id: {arm_id}
          >
          rewards: 0
        >
        """.format(arm_id=time - 1), Feedback()))
