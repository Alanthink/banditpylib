import tempfile

from banditpylib import parse_trials_from_bytes
from banditpylib.arms import BernoulliArm
from banditpylib.bandits import MultiArmedBandit
from banditpylib.learners.mab_learner import EpsGreedy
from .single_player_protocol import SinglePlayerProtocol


class TestSinglePlayer:
  """Test single player protocol"""
  def test_simple_run(self):
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    ordinary_bandit = MultiArmedBandit(arms)
    eps_greedy_learner = EpsGreedy(arm_num=3)
    single_player = SinglePlayerProtocol(bandit=ordinary_bandit,
                                         learners=[eps_greedy_learner])
    temp_file = tempfile.NamedTemporaryFile()
    single_player.play(3, temp_file.name, horizon=10)

    with open(temp_file.name, 'rb') as f:
      # check number of records is 3
      trials = parse_trials_from_bytes(f.read())
      assert len(trials) == 3
