from banditpylib.arms import BernoulliArm
from banditpylib.learners import MaximizeCorrectAnswers, MakeAllAnswersCorrect
from .thresholding_bandit import ThresholdingBandit


class TestThresholdingBandit:
  """Test thresholding bandit"""
  def test_simple_run(self):
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    thres_bandit = ThresholdingBandit(arms=arms, theta=0.5, eps=0)
    thres_bandit.reset()
    assert thres_bandit.regret(MaximizeCorrectAnswers(answers=[0, 1, 1])) == 0
    assert thres_bandit.regret(MakeAllAnswersCorrect(answers=[0, 1, 0])) == 1
