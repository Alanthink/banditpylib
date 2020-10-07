import tempfile

from banditpylib.arms import BernoulliArm
from banditpylib.bandits import OrdinaryBandit
from banditpylib.learners.ordinary_learner import EpsGreedy
from .single_player import SinglePlayerProtocol


class TestSinglePlayer:
  """Test single player protocol"""

  def test_simple_run(self):
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = OrdinaryBandit(arms)
    eps_greedy_learner = EpsGreedy(arm_num=3, horizon=10)
    single_player = SinglePlayerProtocol(bandit=ordinary_bandit,
                                         learners=[eps_greedy_learner])
    temp_file = tempfile.NamedTemporaryFile()
    single_player.play(trials=3, output_filename=temp_file.name)
    with open(temp_file.name, 'r') as f:
      # check number of records is 3
      lines = f.readlines()
      assert len(lines) == 3
