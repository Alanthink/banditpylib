from abc import abstractmethod

from typing import Optional, Union, List, Iterable

from banditpylib.bandits import MultiArmedBandit
from banditpylib.data_pb2 import Arm, CollaborativeActions, Context
from banditpylib.learners import Learner, Goal, IdentifyBestArm


class CollaborativeLearner(Learner):
  """Base class for collaborative learners in the ordinary multi-armed bandit

  This learner aims to identify the best arm with other learners.

  :param int arm_num: number of arms
  :param str name: alias name
  """
  def __init__(self, arm_num: int, name: Optional[str]):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    self.__arm_num = arm_num

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MultiArmedBandit

  @property
  def arm_num(self) -> int:
    """Number of arms"""
    return self.__arm_num

  @abstractmethod
  def complete_round(self):
    """Routine that resets round-dependent variables"""

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Index of the best arm identified by the learner"""

  @property
  def goal(self) -> Goal:
    arm = Arm()
    arm.id = self.best_arm
    return IdentifyBestArm(best_arm=arm)


class CollaborativeMaster(Learner):
  """Base class for master controlling collaborative agents
  in the ordinary multi-armed bandit

  This learner aims to identify the best arm using other learners.

  :param int arm_num: number of arms
  :param int num_agents: number of agents being used
  :param CollaborativeLearner agent_class:
    the class that defines agent behaviour
  :param str name: alias name
  """
  def __init__(self, arm_num: int, num_agents: int,
    agent_class: CollaborativeLearner, name: Optional[str]):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    if num_agents < 1:
      raise ValueError('Number of agents is expected at least 1. Got %d.' %
                       num_agents)
    self.__arm_num = arm_num
    self.__agent_class = agent_class

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MultiArmedBandit

  @property
  def arm_num(self) -> int:
    """Number of arms"""
    return self.__arm_num

  @property
  def agent_class(self) -> CollaborativeLearner:
    """Class of agents being used"""
    return self.__agent_class

  def actions(self, context=None) -> CollaborativeActions:
    """this is not required, only a filler"""
    del context
    return CollaborativeActions()

  @abstractmethod
  def iterable_actions(self, context: Context) \
    -> Iterable[CollaborativeActions]:
    """Iterable function that returns the actions that need feedback"""

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Index of the best arm identified by the learner"""

  @property
  def goal(self) -> Goal:
    arm = Arm()
    arm.id = self.best_arm
    return IdentifyBestArm(best_arm=arm)
