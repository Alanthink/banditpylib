from typing import Optional, List, Tuple, Dict

import numpy as np

from banditpylib.data_pb2 import Feedback, Actions, Context

from .lilucb_heur_collaborative_utils import assign_arms, \
    get_num_pulls_per_round, CentralizedLilUCBHeuristic
from .utils import MABCollaborativeFixedTimeBAIAgent, \
    MABCollaborativeFixedTimeBAIMaster, MABCollaborativeFixedTimeBAILearner


class LilUCBHeuristicAgent(MABCollaborativeFixedTimeBAIAgent):
  """Agent of collaborative learning

  :param int arm_num: number of arms of the bandit
  :param int rounds: number of total rounds allowed
  :param int horizon: total number of pulls allowed
  :param Optional[str] name: alias name
  """

  # Stages within the agent
  UNASSIGNED = "unassigned"
  CENTRALIZED_LEARNING = "centralized_learning"
  LEARNING = "learning"
  COMMUNICATION = "communication"
  TERMINATION = "termination"

  def __init__(self,
               arm_num: int,
               rounds: int,
               horizon: int,
               name: Optional[str] = None):
    super().__init__(name)
    self.__arm_num = arm_num
    self.__rounds = rounds
    self.__horizon = horizon
    self.reset()

  def _name(self) -> str:
    return "lilucb_heuristic_collaborative_agent"

  def reset(self):
    self.__round_index = 0
    self.__stage = self.UNASSIGNED

  def set_input_arms(self, arms: List[int]):
    if self.__stage != self.UNASSIGNED:
      raise Exception("The agent is expected in stage unassigned. Got %s." %
                      self.__stage)

    if self.__round_index == 0:
      if len(arms) > 1:
        self.__use_centralized_algo = True
        self.__num_pulls_per_round = get_num_pulls_per_round(
            rounds=self.__rounds,
            horizon=self.__horizon,
            use_centralized_learning=True)
      else:
        self.__use_centralized_algo = False
        self.__num_pulls_per_round = get_num_pulls_per_round(
            rounds=self.__rounds,
            horizon=self.__horizon,
            use_centralized_learning=False)

    if arms[0] < 0:
      # Terminate since there is only one active arm
      self.__best_arm = arms[1]
      self.__stage = self.TERMINATION
      return

    self.__assigned_arms = arms
    # Maintain empirical informaiton of assigned arms
    self.__assigned_arm_info: Dict[int, Tuple[float, int]] = {}
    for arm_id in arms:
      self.__assigned_arm_info[arm_id] = (0.0, 0)

    if self.__round_index == (self.__rounds - 1):
      # Last round
      self.__best_arm = arms[0]
      self.__stage = self.TERMINATION
    else:
      if self.__round_index == 0 and self.__use_centralized_algo:
        # Confidence of 0.99 suggested in the paper
        self.__central_algo = CentralizedLilUCBHeuristic(
            self.__arm_num, 0.99, np.array(arms))
        self.__central_algo.reset()
        self.__stage = self.CENTRALIZED_LEARNING
      else:
        if len(self.__assigned_arms) > 1:
          raise Exception("Got more than 1 arm in stage learning.")

        self.__arm_to_broadcast = arms[0]
        self.__stage = self.LEARNING

  def actions(self, context: Context = None) -> Actions:
    if self.__stage == self.UNASSIGNED:
      raise Exception("%s: I can\'t act in stage unassigned." % self.name)

    if self.__stage == self.CENTRALIZED_LEARNING:
      if self.__round_index > 0:
        raise Exception("Expected centralized learning in round 0. Got %d." %
                        self.__round_index)

      if self.__central_algo.get_total_pulls(
      ) >= self.__num_pulls_per_round[0]:
        # Early stop the centralized algorithm when it uses more than horizon
        # / 2 pulls.
        self.__stage = self.LEARNING
        self.__arm_to_broadcast = np.random.choice(self.__assigned_arms)
        self.__round_index += 1
        return self.actions()

      if len(self.__assigned_arms) == 1:
        self.__stage = self.LEARNING
        self.__arm_to_broadcast = self.__assigned_arms[0]
        self.__round_index += 1
        return self.actions()

      central_algo_actions = self.__central_algo.actions()
      if not central_algo_actions.arm_pulls:
        # Centralized algorithm terminates before using up horizon / 2 pulls
        self.__stage = self.LEARNING
        self.__arm_to_broadcast = self.__central_algo.best_arm
        self.__round_index += 1
        return self.actions()
      return central_algo_actions
    elif self.__stage == self.LEARNING:
      actions = Actions()
      arm_pull = actions.arm_pulls.add()
      arm_pull.arm.id = self.__arm_to_broadcast
      arm_pull.times = self.__num_pulls_per_round[self.__round_index]
      return actions
    elif self.__stage == self.COMMUNICATION:
      actions = Actions()
      actions.state = Actions.WAIT
      return actions
    else:
      # self.__stage == self.TERMINATION
      actions = Actions()
      actions.state = Actions.STOP
      return actions

  def update(self, feedback: Feedback):
    if self.__stage not in [self.CENTRALIZED_LEARNING, self.LEARNING]:
      raise Exception("%s: I can\'t do update in stage not learning." %
                      self.name)

    for arm_feedback in feedback.arm_feedbacks:
      old_arm_info = self.__assigned_arm_info[arm_feedback.arm.id]
      new_arm_info = (
          (old_arm_info[0] * old_arm_info[1] + sum(arm_feedback.rewards)) /
          (old_arm_info[1] + len(arm_feedback.rewards)),
          old_arm_info[1] + len(arm_feedback.rewards))
      self.__assigned_arm_info[arm_feedback.arm.id] = new_arm_info

    if self.__stage == self.CENTRALIZED_LEARNING:
      self.__central_algo.update(feedback)
    else:
      # self.__stage == self.LEARNING
      self.__stage = self.COMMUNICATION

  @property
  def best_arm(self) -> int:
    if self.__stage != self.TERMINATION:
      raise Exception('%s: I don\'t have an answer yet.' % self.name)
    return self.__best_arm

  def broadcast(self) -> Dict[int, Tuple[float, int]]:
    if self.__stage != self.COMMUNICATION:
      raise Exception('%s: I can\'t broadcast in stage %s.'\
        % (self.name, self.__stage))

    # Complete the current round
    self.__round_index += 1
    self.__stage = self.UNASSIGNED

    message: Dict[int, Tuple[float, int]] = {}
    message[self.__arm_to_broadcast] = self.__assigned_arm_info[
        self.__arm_to_broadcast]
    return message


class LilUCBHeuristicMaster(MABCollaborativeFixedTimeBAIMaster):
  """Master of collaborative learning

  :param int arm_num: number of arms of the bandit
  :param int rounds: number of total rounds allowed
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: number of agents
  :param Optional[str] name: alias name
  """
  def __init__(self,
               arm_num: int,
               rounds: int,
               horizon: int,
               num_agents: int,
               name: Optional[str] = None):
    super().__init__(name)
    self.__arm_num = arm_num
    self.__comm_rounds = rounds - 1
    self.__T = horizon
    self.__num_agents = num_agents

  def _name(self) -> str:
    return "lilucb_heuristic_collaborative_master"

  def reset(self):
    self.__active_arms = list(range(self.__arm_num))

  def initial_arm_assignment(self) -> Dict[int, List[int]]:
    return assign_arms(self.__active_arms, list(range(self.__num_agents)))

  def elimination(
      self, messages: Dict[int, Dict[int,
                                     Tuple[float,
                                           int]]]) -> Dict[int, List[int]]:

    aggregate_messages: Dict[int, Tuple[float, int]] = {}
    for agent_id in messages.keys():
      message_from_agent = messages[agent_id]
      for arm_id in message_from_agent:
        if arm_id not in aggregate_messages:
          aggregate_messages[arm_id] = (0.0, 0)
        arm_info = message_from_agent[arm_id]
        new_pulls = aggregate_messages[arm_id][1] + arm_info[1]
        new_em_mean_reward = (aggregate_messages[arm_id][0] * \
            aggregate_messages[arm_id][1] + arm_info[0] * arm_info[1]) \
            / new_pulls
        aggregate_messages[arm_id] = (new_em_mean_reward, new_pulls)

    accumulated_arm_ids = np.array(list(aggregate_messages.keys()))
    accumulated_em_mean_rewards = np.array(
        list(map(lambda x: aggregate_messages[x][0],
                 aggregate_messages.keys())))

    # Elimination
    confidence_radius = np.sqrt(
        self.__comm_rounds *
        np.log(200 * self.__num_agents * self.__comm_rounds) /
        (self.__T * max(1, self.__num_agents / len(self.__active_arms))))
    highest_em_reward = np.max(accumulated_em_mean_rewards)
    self.__active_arms = list(
        accumulated_arm_ids[accumulated_em_mean_rewards >= (
            highest_em_reward - 2 * confidence_radius)])

    return assign_arms(self.__active_arms, list(messages.keys()))


class LilUCBHeuristicCollaborative(MABCollaborativeFixedTimeBAILearner):
  """Colaborative learner using lilucb heuristic as centralized policy

  :param int num_agents: number of agents
  :param int arm_num: number of arms of the bandit
  :param int rounds: number of total rounds allowed
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param Optional[str] name: alias name
  """
  def __init__(self,
               num_agents: int,
               arm_num: int,
               rounds: int,
               horizon: int,
               name: Optional[str] = None):
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    if rounds <= 2:
      raise ValueError('Number of rounds is expected at least 2. Got %d.' %
                       rounds)

    if horizon <= rounds - 1:
      raise ValueError(
          'Horizon is expected at least total rounds minus one. Got %d.' %
          horizon)

    super().__init__(agent=LilUCBHeuristicAgent(arm_num=arm_num,
                                                rounds=rounds,
                                                horizon=horizon),
                     master=LilUCBHeuristicMaster(arm_num=arm_num,
                                                  rounds=rounds,
                                                  horizon=horizon,
                                                  num_agents=num_agents),
                     num_agents=num_agents,
                     name=name)

  def _name(self) -> str:
    return 'lilucb_heuristic_collaborative'
