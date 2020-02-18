import time

from abc import ABC, abstractmethod

from absl import flags

FLAGS = flags.FLAGS

__all__ = ['Protocol', 'current_time']


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Protocol(ABC):
  """Abstract protocol class"""

  @property
  @abstractmethod
  def type(self):
    pass

  @abstractmethod
  def play(self, bandits, players, output_file, pars):
    pass
