import tempfile

from banditpylib.arms import BernoulliArm
from banditpylib.bandits import MultiArmedBandit
from banditpylib.learners.mab_learner import EpsGreedy
from .single_player import SinglePlayerProtocol
from .utils import parse_trials_data


class TestSinglePlayer:
  """Test single player protocol"""
  def test_simple_run(self):
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = MultiArmedBandit(arms)
    eps_greedy_learner = EpsGreedy(arm_num=3)
    single_player = SinglePlayerProtocol(bandit=ordinary_bandit,
                                         learners=[eps_greedy_learner],
                                         horizon=10)
    temp_file = tempfile.NamedTemporaryFile()
    single_player.play(trials=3, output_filename=temp_file.name)

    with open(temp_file.name, 'rb') as f:
      # check number of records is 3
      trials_data = parse_trials_data(f.read())
      assert len(trials_data) == 3
