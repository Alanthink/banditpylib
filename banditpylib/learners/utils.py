from abc import ABC, abstractmethod


class Learner(ABC):
  """
  Abstract class for a learner.

  Before a game runs, a learner should be initialized with :func:`reset`.
  """

  def __init__(self, pars):
    """Learner initialization

    Args:
      pars (dict): has form of

        .. code-block:: yaml

          {
            # Name appeared in the output figure. Default
            # value `self._name`.
            "name": string,
          }
    """
    self.__name = pars['name'] if 'name' in pars else None

  @property
  def name(self):
    """str: learner name"""
    if self.__name:
      return self.__name
    return self._name

  @property
  @abstractmethod
  def _name(self):
    """Default name of the learner"""

  @property
  @abstractmethod
  def goal(self):
    """str: goal of the learner"""

  @abstractmethod
  def reset(self, bandit, stop_cond):
    """Reset self.
    This function should be called before the start of the game.

    Args:
      bandit (object): environment
      stop_cond (dict): stop conditions
    """
