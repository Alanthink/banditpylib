from typing import Optional, Union, List, Tuple, Dict
from copy import deepcopy as dcopy
from abc import ABC, abstractmethod

from banditpylib.bandits import MultiArmedBandit
from banditpylib.data_pb2 import Arm, Feedback, Actions, Context
from banditpylib.learners import Goal, IdentifyBestArm, Learner


class CollaborativeBAIAgent(ABC):
  r"""One individual agent

  This agent aims to identify the best arm with other agents.

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """Name of the agent"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default agent name
    """

  @abstractmethod
  def reset(self):
    """Reset the agent

    .. warning::
      This function should be called before the start of each game.
    """

  @abstractmethod
  def set_input_arms(self, arms: List[int]):
    """Assign a set of arms to the agent

    Args:
      arms: arm indices that have been assigneds
    """

  @abstractmethod
  def actions(self, context: Context) -> Actions:
    """Actions of the agent

    Args:
      context: contextual information about the bandit environment

    Returns:
      actions to take
    """

  @abstractmethod
  def update(self, feedback: Feedback):
    """Update the agent

    Args:
      feedback: feedback returned by the bandit environment after
        :func:`actions` is executed
    """

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Arm that the agent chose (could be None)"""

  @abstractmethod
  def broadcast(self) -> Dict[int, Tuple[float, int]]:
    """Broadcasts information learnt in the current round

    Returns: (a dict of)
      arm ids used in learning
      Tuple[corresponding average reward seen,
        number of pulls used to deduce average]
    """


class CollaborativeBAIMaster(ABC):
  r"""Master that handles arm assignment and elimination

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """Name of the master"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default master name
    """

  @abstractmethod
  def reset(self):
    """Reset the master

    .. warning::
      This function should be called before the start of each game.
    """

  @abstractmethod
  def initial_arm_assignment(self) -> Dict[int, List[int]]:
    """The arm assignment for the first round

    Returns:
      dictionary of arm assignment per agent for all agents
    """

  @abstractmethod
  def elimination(
      self, agent_ids: List[int],
      messages: Dict[int, Dict[int, Tuple[float,
                                          int]]]) -> Dict[int, List[int]]:
    """Update the set of active arms based on some criteria
    and return arm assignment

    Args:
      agent_ids: list of agents that will be assigned arms
      messages: dict of messages broadcasted from agents

    Returns:
      dictionary of arm assignment per agent
    """


class CollaborativeBAILearner(Learner):
  """Learner that puts the agents and master together

  :param CollaborativeAgent agent: one instance of an agent
  :param CollaboratveMaster master: instance of the master
  :param int num_agents: total number of agents involved
  :param Optional[str] name: alias name
  """
  def __init__(self,
               agent: CollaborativeBAIAgent,
               master: CollaborativeBAIMaster,
               num_agents: int,
               name: Optional[str] = None):
    super().__init__(name)
    self.__agents = []
    for _ in range(num_agents):
      self.__agents.append(dcopy(agent))
    self.__master = master

  def _name(self) -> str:
    return 'collaborative_learner'

  def reset(self):
    for agent in self.__agents:
      agent.reset()
    self.__master.reset()

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MultiArmedBandit

  def get_agents(self) -> List[CollaborativeBAIAgent]:
    """Involved agents"""
    return self.__agents

  def get_master(self) -> CollaborativeBAIMaster:
    """Controlling Master"""
    return self.__master

  @property
  def goal(self) -> Goal:
    arm = Arm()
    best_arm = self.__agents[0].best_arm
    for agent in self.__agents[1:]:
      if best_arm != agent.best_arm:
        best_arm = -1  # implies regret of 1
        break
    arm.id = best_arm
    return IdentifyBestArm(best_arm=arm)
