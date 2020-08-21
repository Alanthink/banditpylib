from abc import ABC, abstractmethod


class Learner(ABC):
  """Learner class

  Before a game runs, a learner should be initialized with :func:`reset`.
  """

  @property
  def name(self) -> str:
    """
    Return:
      name of the learner
    """

  @abstractmethod
  def reset(self):
    """Reset of the learner.

    This function should be called before the start of the game
    """

  @abstractmethod
  def actions(self, context=None):
    """Actions of the learner for one round

    Args:
      context: context of the bandit environment

    Return:
      actions: actions to take
    """

  @abstractmethod
  def update(self, feedback):
    """Learner update

    Args:
      feedback: feedback of the bandit environment
    """
