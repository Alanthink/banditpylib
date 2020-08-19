from arms import BernoulliArm
from bandits import OrdinaryBandit
from .eps_greedy import EpsGreedy


class TestEpsGreedy:
  """Test epsilon greedy policy"""

  def test_simple_run(self):
    means = [0, 0.5, 0.7, 1]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = OrdinaryBandit(arms)
    eps_greedy_learner = EpsGreedy(arm_num=4, horizon=10)

    ordinary_bandit.reset()
    eps_greedy_learner.reset()
    rounds = 0
    while True:
      actions = eps_greedy_learner.actions()
      if not actions:
        break
      feedback = ordinary_bandit.feed(actions)
      eps_greedy_learner.update(feedback)
      rounds += 1
    assert rounds == 10, 'Number of rounds %d is not equal to 10!' % rounds
