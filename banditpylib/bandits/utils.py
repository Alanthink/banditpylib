from abc import ABC, abstractmethod

from banditpylib.data_pb2 import Context, Actions, Feedback
from banditpylib.learners import Goal


class Bandit(ABC):
  """Abstract class for bandit environments"""
  @property
  @abstractmethod
  def name(self) -> str:
    """Bandit name"""

  @abstractmethod
  def reset(self):
    """Reset the bandit environment

    .. warning::
      This function should be called before the start of the game.
    """

  @property
  @abstractmethod
  def context(self) -> Context:
    """Contextual information about the bandit environment"""

  @abstractmethod
  def feed(self, actions: Actions) -> Feedback:
    """
    Args:
      actions: actions for the bandit environment to execute

    Returns:
      feedback after `actions` are executed
    """

  @abstractmethod
  def regret(self, goal: Goal) -> float:
    """
    Args:
      goal: goal of the learner

    Returns:
      regret of the learner
    """
