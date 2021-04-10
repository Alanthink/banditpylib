from banditpylib.arms import BernoulliArm
from banditpylib.learners import MaxCorrectAnswers, AllCorrect
from .thres_bandit import ThresholdingBandit


class TestThresholdingBandit:
  """Test thresholding bandit"""
  def test_simple_run(self):
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    thres_bandit = ThresholdingBandit(arms=arms, theta=0.5, eps=0)
    thres_bandit.reset()
    assert thres_bandit.regret(MaxCorrectAnswers(answers=[0, 1, 1])) == 0
    assert thres_bandit.regret(AllCorrect(answers=[0, 1, 0])) == 1
