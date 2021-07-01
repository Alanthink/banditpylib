from typing import List, cast
from copy import deepcopy as dcopy

import numpy as np
from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial, Actions
from banditpylib.learners import Learner
from banditpylib.learners.collaborative_learner import CollaborativeBAILearner
from .utils import Protocol


class CollaborativeLearningProtocol(Protocol):
  """Collaborative learning protocol :cite:'arXiv:1904.03293'

  This protocol is used to simulate the multi-agent game
  as discussed in the paper. It runs in rounds. During each round,
  the protocol runs the following steps in sequence:

  * fetch the state of the environment and ask each learner for actions;
  * send the actions to the enviroment for execution;
  * update each learner with the corresponding feedback of the environment;
  * repeat the above steps until every agent enters the WAIT or STOP state;
  * if there is at least one agent in WAIT state, then receive information
    broadcasted from every waiting agent, and use it to decide
    the parameters of the next round.

  The simulation stopping criteria is:

  * every agent enters STOP state;

  Algorithm is guaranteed to stop before total number of time-steps
  achieve `horizon`.

  :param Bandit bandit: bandit environment
  :param List[CollaborativeBAILearner] learners: learners that will be compared

  .. note::
    During a timestep, a learner may want to perform multiple actions, which is
    so-called batched learner. In this case, eah action counts as a timestep
    used.
  """
  def __init__(self,
               bandit: Bandit, learners: List[CollaborativeBAILearner]):
    super().__init__(bandit=bandit, learners=cast(List[Learner], learners))

  @property
  def name(self) -> str:
    return 'collaborative_learning_protocol'

  def _one_trial(self, random_seed: int, debug: bool) -> bytes:
    if debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    # initialization
    current_learner = cast(CollaborativeBAILearner, self.current_learner)
    current_learner.reset()
    agents = current_learner.get_agents()
    bandits = []
    master = current_learner.get_master()
    for _ in range(len(agents)):
      bandits.append(dcopy(self.bandit))
      bandits[-1].reset()

    trial = Trial()
    trial.bandit = self.bandit.name
    trial.learner = current_learner.name

    round_num, total_pulls_used = 0, 0

    # stages of the algorithms
    stopped_agents = [False] * len(agents) # terminated agents
    while True:
      # using only non-stopped agents
      running_agents = []
      for i, agent in enumerate(agents):
        if not stopped_agents[i]:
          running_agents.append(agent)
      if len(running_agents) == 0:
        break

      arms_assign_list, num_active_arms = master.get_assigned_arms(
        len(running_agents))
      for i, agent in enumerate(running_agents):
        agent.assign_arms(arms_assign_list[i], num_active_arms)

      waiting_agents = [False] * len(agents) # waiting for communication
      # preparation and learning
      for i, agent in enumerate(agents):
        while True:
          actions = agent.actions(bandits[i].context)
          if actions.state == Actions.WAIT:
            waiting_agents[i] = True
            break
          elif actions.state == Actions.STOP:
            stopped_agents[i] = True
            break
          else:
            feedback = bandits[i].feed(actions)
            agent.update(feedback)

      # stop if all agents are in STOP
      if sum(stopped_agents) == len(agents):
        break

      # otherwise atleat one agent is in WAIT
      # communication and aggregation
      arm_ids, em_mean_rewards, pulls_used_list = [], [], []
      for i, agent in enumerate(agents):
        if waiting_agents[i]:
          arm_ids_used, em_mean_rewards_seen, pulls_used = agent.broadcast()
          pulls_used_list.append(pulls_used)
          for j, arm_id in enumerate(arm_ids_used):
            if arm_id is not None:
              arm_ids.append(arm_id)
              em_mean_rewards.append(em_mean_rewards_seen[j])

      # if arm_ids list is empty (agents broadcasted nothing)
      if len(arm_ids)==0:
        # return after adding data
        result = trial.results.add()
        result.rounds = round_num
        result.total_actions = total_pulls_used
        result.regret = 1
        return trial.SerializeToString()

      # send info to master for elimination
      master.elimination(arm_ids, em_mean_rewards)

      for i, agent in enumerate(agents):
        agent.complete_round()
        if agent.stage == "termination":
          stopped_agents[i] = True

      round_num += 1
      total_pulls_used += max(pulls_used_list)

    # add data when algorithm stops running
    result = trial.results.add()
    result.rounds = round_num
    result.total_actions = total_pulls_used
    result.regret = self.bandit.regret(
      current_learner.goal)

    return trial.SerializeToString()
