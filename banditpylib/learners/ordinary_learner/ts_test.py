from unittest.mock import MagicMock

from .ts import ThompsonSampling


class TestThompsonSampling:
  """Test thompson sampling policy"""
  def test_simple_run(self):
    ts_learner = ThompsonSampling(arm_num=4, horizon=10)
    ts_learner.reset()
    ts_learner.actions_from_beta_prior = MagicMock(return_value=1)
    # always pull arm 1
    assert ts_learner.actions() == [(1, 1)]
