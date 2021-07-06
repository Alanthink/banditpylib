from typing import List, cast, Dict, Tuple
from copy import deepcopy as dcopy

import numpy as np
from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial, Actions
from banditpylib.learners import Learner
from banditpylib.learners.collaborative_learner import CollaborativeBAILearner
from .utils import Protocol


class CollaborativeLearningProtocol(Protocol):
  """Collaborative learning protocol :cite:`tao2019collaborative`

  This protocol is used to simulate the multi-agent game
  as discussed in the paper. It runs in rounds. During each round,
  the protocol runs the following steps in sequence:

  * fetch the state of the environment and ask each learner for actions;
  * send the actions to the enviroment for execution;
  * update each learner with the corresponding feedback of the environment;
  * repeat the above steps until every agent enters the WAIT or STOP state;
  * if there is at least one agent in WAIT state, then receive information
    broadcasted from every waiting agent and send them to master to decide
    arm assignment of next round.

  The simulation stopping criteria is:

  * every agent enters STOP state;

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

    communication_rounds, total_pulls = 0, 0
    active_agent_ids = list(range(len(agents)))
    agent_arm_assignment = master.initial_arm_assignment()
    for agent_id in agent_arm_assignment:
      agents[agent_id].set_input_arms(agent_arm_assignment[agent_id])

    while True:
      max_pulls = 0
      agent_in_wait_ids = []

      # preparation and learning
      for agent_id in active_agent_ids:
        agent = agents[agent_id]
        pulls = 0
        while True:
          actions = agent.actions(bandits[agent_id].context)
          for arm_pull in actions.arm_pulls:
            pulls += arm_pull.times

          if actions.state == Actions.DEFAULT_NORMAL:
            feedback = bandits[agent_id].feed(actions)
            agent.update(feedback)
          elif actions.state == Actions.WAIT:
            agent_in_wait_ids.append(agent_id)
            break
          else: # actions.state == Actions.STOP
            break
        max_pulls = max(max_pulls, pulls)
      total_pulls += max_pulls

      # stop if all agents are in STOP <=> no agents in WAIT
      if not agent_in_wait_ids:
        break

      # communication and aggregation
      # key is arm id and target is a dict broadcasted by agent
      accumulated_messages: Dict[int, Dict[int, Tuple[float, int]]] = {}
      for agent_id in agent_in_wait_ids:
        agent = agents[agent_id]

        # key is arm_id, target is Tuple[em_mean_reward, pulls]
        # empty dict => arm_id was None
        message_from_agent = agent.broadcast()
        accumulated_messages[agent_id] = message_from_agent

      # Send info to master for elimination to get arm assignment for next round
      # agent_arm_assignment: key is agent_id, value is a list storing arm ids
      # assigned to this agent
      agent_arm_assignment = master.elimination(
        agent_in_wait_ids, accumulated_messages)
      for agent_id in agent_arm_assignment:
        agents[agent_id].set_input_arms(agent_arm_assignment[agent_id])
      communication_rounds += 1

    # add data when algorithm stops running
    result = trial.results.add()
    result.rounds = communication_rounds
    result.total_actions = total_pulls
    result.regret = self.bandit.regret(
      current_learner.goal)

    return trial.SerializeToString()
