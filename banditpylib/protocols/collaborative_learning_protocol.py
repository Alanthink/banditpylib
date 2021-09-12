from typing import List, cast, Dict, Tuple
from copy import deepcopy as dcopy

import numpy as np
from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial, Actions
from banditpylib.learners import Learner, CollaborativeLearner
from .utils import Protocol


class CollaborativeLearningProtocol(Protocol):
  """Collaborative learning protocol :cite:`tao2019collaborative`

  This class defines the communication protocol for the collaborative learning
  multi-agent game as discussed in the reference paper. The game runs in
  rounds. During each round, the protocol runs the following steps in sequence:

  - For each agent,

    * fetch the state of the corresponding bandit environment and ask the agent
      for actions;
    * send the actions to the bandit environment for execution;
    * update the agent with the feedback of the bandit environment;
    * repeat the above steps until the agent enters the `WAIT` or `STOP` state.

  - If there is at least one agent in `WAIT` state, then fetch information
    broadcasted from every waiting agent and send them to master to decide
    arm assignment of next round. Otherwise, stop the game.

  :param Bandit bandit: bandit environment
  :param List[CollaborativeLearner] learners: learners that will be compared
    with

  .. note::
    Each agent interacts with an independent bandit environment.

  .. note::
    Each action counts as a timestep. The time (or sample) complexity equals to
    the maximum number of pulls across different agents.

  .. note::
    According to the protocol, number of rounds always equals to number of
    communication rounds plus one.
  """
  def __init__(self, bandit: Bandit, learners: List[CollaborativeLearner]):
    super().__init__(bandit=bandit, learners=cast(List[Learner], learners))

  @property
  def name(self) -> str:
    return 'collaborative_learning_protocol'

  def _one_trial(self, random_seed: int) -> bytes:
    if self._debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    # Initialization
    current_learner = cast(CollaborativeLearner, self._current_learner)
    current_learner.reset()
    agents = current_learner.agents
    bandits = []
    master = current_learner.master
    for _ in range(len(agents)):
      bandits.append(dcopy(self._bandit))
      bandits[-1].reset()

    trial = Trial()
    trial.bandit = self._bandit.name
    trial.learner = current_learner.name

    communication_rounds, total_pulls = 0, 0
    active_agent_ids = list(range(len(agents)))
    agent_arm_assignment = master.initial_arm_assignment()
    for agent_id in agent_arm_assignment:
      agents[agent_id].set_input_arms(agent_arm_assignment[agent_id])

    while True:
      max_pulls = 0
      agent_in_wait_ids = []

      # Preparation and learning
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
          else:  # actions.state == Actions.STOP
            break
        max_pulls = max(max_pulls, pulls)
      total_pulls += max_pulls

      # Stop if all agents are in STOP states which is equivalent to that no
      # agents are in WAIT states
      if not agent_in_wait_ids:
        break

      # Communication and aggregation
      # Key is agent id and target is a dict storing information broadcasted by
      # the agent
      accumulated_messages: Dict[int, Dict[int, Tuple[float, int]]] = {}
      for agent_id in agent_in_wait_ids:
        agent = agents[agent_id]

        message_from_agent = agent.broadcast()
        accumulated_messages[agent_id] = message_from_agent

      # Send info to master for elimination to get arm assignment for next round
      agent_arm_assignment = master.elimination(accumulated_messages)
      for agent_id in agent_arm_assignment:
        agents[agent_id].set_input_arms(agent_arm_assignment[agent_id])
      communication_rounds += 1

    # Add simulation results when algorithm stops running
    result = trial.results.add()
    result.rounds = communication_rounds
    result.total_actions = total_pulls
    result.regret = self._bandit.regret(current_learner.goal)

    return trial.SerializeToString()
