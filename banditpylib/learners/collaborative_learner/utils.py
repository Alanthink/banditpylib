from abc import abstractmethod

from typing import Optional, Union, List, Tuple

from banditpylib.bandits import MultiArmedBandit
from banditpylib.data_pb2 import Arm
from banditpylib.learners import Learner, Goal, IdentifyBestArm

from .collaborative_agent import CollaborativeMaster

class CollaborativeLearner(Learner):
  """Base class for collaborative learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with other learners.

  :param int arm_num: number of arms
  :param int rounds: number of rounds of communication allowed
    (this agent uses one more)
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: total number of agents involved
  :param str name: alias name
  """
  def __init__(self, arm_num: int, rounds: int, horizon: int,
    num_agents: int, master:CollaborativeMaster, name: Optional[str]):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    self.__arm_num = arm_num
    self.__rounds = rounds
    self.__horizon = horizon
    self.__num_agents = num_agents
    self.__master = master

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MultiArmedBandit

  @property
  def arm_num(self) -> int:
    """Number of arms"""
    return self.__arm_num

  @property
  def rounds(self) -> int:
    """Number of rounds"""
    return self.__rounds

  @property
  def horizon(self) -> int:
    """Time horizon"""
    return self.__horizon

  @property
  def num_agents(self) -> int:
    """Number of agents"""
    return self.__num_agents

  @property
  def master(self) -> CollaborativeMaster:
    """Controlling Master"""
    return self.__master

  @abstractmethod
  def complete_round(self):
    """Routine that resets round-dependent variables"""

  @abstractmethod
  def broadcast(self) -> Tuple[int, float, int]:
    """Broadcast information learnt in the current round"""

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Index of the best arm identified by the learner"""

  @property
  def goal(self) -> Goal:
    arm = Arm()
    arm.id = self.best_arm
    return IdentifyBestArm(best_arm=arm)
