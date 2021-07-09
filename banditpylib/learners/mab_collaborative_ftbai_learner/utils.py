from typing import Optional, Union, List, cast
from abc import abstractmethod
import random

from banditpylib.bandits import MultiArmedBandit
from banditpylib.data_pb2 import Arm
from banditpylib.learners import Goal, IdentifyBestArm, \
  CollaborativeLearner, CollaborativeAgent, CollaborativeMaster


class MABCollaborativeFixedTimeBAIAgent(CollaborativeAgent):
  """Abstract agent identify the best arm with fixed time

  This agent aims to identify the best arm with other agents.

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    super().__init__(name=name)

  @property
  @abstractmethod
  def best_arm(self) -> int:
    """Arm chosen by the agent"""


class MABCollaborativeFixedTimeBAIMaster(CollaborativeMaster):
  """Abstract master to identify the best arm with fixed time

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    super().__init__(name=name)


class MABCollaborativeFixedTimeBAILearner(CollaborativeLearner):
  """Learner that puts the agents and master together

  :param CollaborativeAgent agent: one instance of an agent
  :param CollaboratveMaster master: instance of the master
  :param int num_agents: total number of agents involved
  :param Optional[str] name: alias name
  """
  def __init__(self,
               agent: MABCollaborativeFixedTimeBAIAgent,
               master: MABCollaborativeFixedTimeBAIMaster,
               num_agents: int,
               name: Optional[str] = None):
    super().__init__(agent=cast(CollaborativeAgent, agent),
                     master=cast(CollaborativeMaster, master),
                     num_agents=num_agents,
                     name=name)

  def _name(self) -> str:
    return 'collaborative_learner'

  @property
  def running_environment(self) -> Union[type, List[type]]:
    return MultiArmedBandit

  @property
  def goal(self) -> Goal:
    arm = Arm()
    best_arms: List[int] = []
    for agent in self.agents:
      best_arms.append(cast(MABCollaborativeFixedTimeBAIAgent, agent).best_arm)
    arm.id = random.choice(best_arms)
    return IdentifyBestArm(best_arm=arm)
