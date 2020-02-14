from abc import ABC, abstractmethod

__all__ = ['Protocol']


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)

class Protocol(ABC):
  """Abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  @abstractmethod
  def init(self):
    pass

  @abstractmethod
  def _one_trial(self,seed):
  	pass

  @abstractmethod
  def _play_round(self):
    """
    Input:
    Output:
      reward or a list of rewards
    """
    pass

  @abstractmethod
  def regret(self, rewards):
    pass