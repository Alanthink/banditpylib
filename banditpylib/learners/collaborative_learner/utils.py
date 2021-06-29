from typing import Optional, Union, List, Tuple
from copy import deepcopy as dcopy
from abc import ABC, abstractmethod

from banditpylib.bandits import MultiArmedBandit
from banditpylib.data_pb2 import Arm, Feedback, Actions, Context
from banditpylib.learners import Goal, IdentifyBestArm

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
      default learner name
    """

  @abstractmethod
  def reset(self):
    """Reset the agent

    .. warning::
      This function should be called before the start of each game.
    """

  @abstractmethod
  def complete_round(self):
    """Update the round-local variables"""

  @abstractmethod
  def assign_arms(self, arms: List[int], num_active_arms: int):
    """Assign a set of arms to the agent

    Args:
      * arms: arm indices that have been assigned
      * total number of active arms
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

  @property
  def stage(self) -> str:
    """Stage of the agent"""

  @abstractmethod
  def broadcast(self) -> Tuple[int, float, int]:
    """Broadcasts information learnt in the current round

    Returns: (in a tuple)
      * arm used in learning
      * average reward seen
      * pulls used in the current round
    """


class CollaborativeBAIMaster(ABC):
  r"""Master that handles arm assignment and elimination

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
      default learner name
    """

  @abstractmethod
  def reset(self):
    """Reset the master

    .. warning::
      This function should be called before the start of each game.
    """

  @abstractmethod
  def get_assigned_arms(self, num_running_agents: int) ->\
    Tuple[List[List[int]], int]:
    """Assign arms to non-terminated agents

    Args:
      num_running_agents: number of running agents

    Returns:
      * list of set of assigned arms per agent
      * number of active arms
    """

  @abstractmethod
  def elimination(self, i_l_r_list: List[int], p_l_r_list: List[float]):
    """Update the set of active arms based on some criteria

    Args:
      i_l_r_list: list of arm indexes used by the agents in learning
      p_l_r_list: emperical means seen by the agents
    """

  @property
  @abstractmethod
  def active_arms(self):
    """Arm indices that haven't been eliminated"""


class CollaborativeBAILearner():
  """Learner that puts the agents and master together

  :param CollaborativeAgent agent: one instance of an agent
  :param CollaboratveMaster master: instance of the master
  :param int num_agents: total number of agents involved
  :param Optional[str] name: alias name
  """
  def __init__(self, agent: CollaborativeBAIAgent,
    master: CollaborativeBAIMaster, num_agents: int,
    name: Optional[str] = None):
    self.__agents = []
    for _ in range(num_agents):
      self.__agents.append(dcopy(agent))
    self.__master = master
    self.__name = name

  @property
  def name(self) -> str:
    if self.__name is None:
      return 'collaborative_learner'
    return self.__name

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MultiArmedBandit

  @property
  def agents(self) -> List[CollaborativeBAIAgent]:
    """Involved agents"""
    return self.__agents

  @property
  def master(self) -> CollaborativeBAIMaster:
    """Controlling Master"""
    return self.__master

  def agent_goal(self, index) -> Goal:
    if index not in range(len(self.__agents)):
      raise ValueError("Index expected n [0, %d), got %d" %\
        (len(self.__agents), index))
    arm = Arm()
    arm.id = self.__agents[index].best_arm
    return IdentifyBestArm(best_arm=arm)
