from banditpylib.learners import Learner


# pylint: disable=W0223
class OrdinaryLearner(Learner):
  """Base class for learners in the ordinary multi-armed bandit
  """
  def __init__(self, arm_num: int, horizon: int, name: str):
    """
    Args:
      arm_num: number of arms
      horizon: total number of time steps
      name: alias name
    """
    super().__init__(name)
    if arm_num <= 2:
      raise Exception('Number of arms %d is less then 3!' % arm_num)
    self.__arm_num = arm_num
    if horizon < arm_num:
      raise Exception('Horizon %d is less than number of arms %d!' % \
          (horizon, arm_num))
    self.__horizon = horizon

  def arm_num(self) -> int:
    """
    Returns:
      number of arms
    """
    return self.__arm_num

  def horizon(self) -> int:
    """
    Returns:
      horizon of the game
    """
    return self.__horizon

  def set_horizon(self, horizon: int):
    """Reset the horizon

    Args:
      horizon: new horizon of the game
    """
    self.__horizon = horizon

  def regret(self, bandit) -> float:
    """
    Args:
      bandit: bandit environment

    Returns:
      regret compared with the optimal policy
    """
    return bandit.regret()
