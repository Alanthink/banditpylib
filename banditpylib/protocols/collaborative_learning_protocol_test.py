import tempfile

from banditpylib import parse_trials_from_bytes
from banditpylib.arms import GaussianArm
from banditpylib.bandits import MultiArmedBandit
from banditpylib.learners.mab_collaborative_ftbai_learner import (
  LilUCBHeuristicCollaborative)
from .collaborative_learning_protocol import CollaborativeLearningProtocol


class TestCollaborativeLearning:
  """Test collaborative learning protocol"""
  def test_simple_run(self):
    means = [0.3, 0.5, 0.7]
    arms = [GaussianArm(mu=mean, std=1) for mean in means]
    bandit = MultiArmedBandit(arms=arms)
    lil_ucb_collaborative_learner = LilUCBHeuristicCollaborative(num_agents=3,
      arm_num=len(arms), rounds=4, horizon=100)
    collaborative_learner = CollaborativeLearningProtocol(
      bandit=bandit, learners=[lil_ucb_collaborative_learner])
    temp_file = tempfile.NamedTemporaryFile()
    collaborative_learner.play(trials=3, output_filename=temp_file.name)

    with open(temp_file.name, 'rb') as f:
      # check number of records is 3
      trials = parse_trials_from_bytes(f.read())
      assert len(trials) == 3
