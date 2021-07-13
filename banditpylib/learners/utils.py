from abc import ABC, abstractmethod
from copy import deepcopy as dcopy
from typing import Optional, List, Union, Dict, Tuple

from banditpylib.data_pb2 import Context, Arm, Actions, Feedback


class Goal(ABC):
  """Abstract class for the goal of a learner"""
  @property
  @abstractmethod
  def name(self) -> str:
    """Name of the goal"""


class IdentifyBestArm(Goal):
  """Best arm identification

  :param Arm best_arm: best arm identified by the learner
  """
  def __init__(self, best_arm: Arm):
    self.__best_arm = best_arm

  @property
  def name(self) -> str:
    return 'best_arm_id'

  @property
  def best_arm(self) -> Arm:
    return self.__best_arm


class MaximizeTotalRewards(Goal):
  """Reward maximization"""
  @property
  def name(self) -> str:
    return 'reward_maximization'


class MaximizeCorrectAnswers(Goal):
  """Maximize correct answers

  This is used by thresholding bandit learners.

  :param List[int] answers: answers obtained by the learner
  """
  def __init__(self, answers: List[int]):
    self.__answers = answers

  @property
  def name(self) -> str:
    return 'max_correct_answers'

  @property
  def answers(self) -> List[int]:
    return self.__answers


class MakeAllAnswersCorrect(Goal):
  """Make all answers correct

  This is used by thresholding bandit learners.

  :param List[int] answers: answers obtained by the learner
  """
  def __init__(self, answers: List[int]):
    self.__answers = answers

  @property
  def name(self) -> str:
    return 'make_all_correct'

  @property
  def answers(self) -> List[int]:
    return self.__answers


class Learner(ABC):
  """Abstract class for learners

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """Name of the learner"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default learner name
    """

  @abstractmethod
  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """

  @property
  @abstractmethod
  def running_environment(self) -> Union[type, List[type]]:
    """Type of bandit environment the learner plays with"""

  @property
  @abstractmethod
  def goal(self) -> Goal:
    """Goal of the learner"""


class SinglePlayerLearner(Learner):
  """Abstract class for single player learners

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    super().__init__(name)

  @abstractmethod
  def actions(self, context: Context) -> Actions:
    """Actions of the learner

    Args:
      context: contextual information about the bandit environment

    Returns:
      actions to take
    """

  @abstractmethod
  def update(self, feedback: Feedback):
    """Update the learner

    Args:
      feedback: feedback returned by the bandit environment after
        :func:`actions` is executed
    """


class CollaborativeAgent(ABC):
  r"""Abstract class for collaborative agents

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
      arms: arm indices that have been assigned
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

  @abstractmethod
  def broadcast(self) -> Dict[int, Tuple[float, int]]:
    """Broadcast information learnt in the current round

    Returns:
      arm ids, corresponding average rewards seen, and numbers of pulls used to
        deduce average rewards
    """


class CollaborativeMaster(ABC):
  r"""Abstract class for collaborative masters that handle arm assignment and
  elimination

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
    """The arm assignment before the first round

    Returns:
      arm assignment per agent for all agents
    """

  @abstractmethod
  def elimination(
      self, messages: Dict[int, Dict[int,
                                     Tuple[float,
                                           int]]]) -> Dict[int, List[int]]:
    """Update the set of active arms based on some criteria and return arm
    assignment

    Args:
      messages: dict of messages broadcasted from agents, where key is agent_id

    Returns:
      arm assignment per agent
    """


class CollaborativeLearner(Learner):
  """Abstract class for collaborative learners

  :param CollaborativeAgent agent: one instance of a collaborative agent
  :param CollaboratveMaster master: instance of a collaborative master
  :param int num_agents: total number of agents involved
  :param Optional[str] name: alias name
  """
  def __init__(self,
               agent: CollaborativeAgent,
               master: CollaborativeMaster,
               num_agents: int,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.__agents = []
    for _ in range(num_agents):
      self.__agents.append(dcopy(agent))
    self.__master = master

  def reset(self):
    for agent in self.__agents:
      agent.reset()
    self.__master.reset()

  @property
  def agents(self) -> List[CollaborativeAgent]:
    """Involved agents"""
    return self.__agents

  @property
  def master(self) -> CollaborativeMaster:
    """Controlling master"""
    return self.__master
