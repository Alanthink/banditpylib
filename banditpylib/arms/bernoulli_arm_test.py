from .bernoulli_arm import BernoulliArm


class TestBernoulliArm:
  """Test Bernoulli arm"""
  def test_reward_range(self):
    ber_arm = BernoulliArm(0.5)
    set_of_rewards = set(ber_arm.pull(10))
    assert set_of_rewards == set({0, 1}), \
        "Some rewards are out of range {0, 1}!"
