from absl import logging

from bandits import ordinarybandit
from .utils import Protocol

__all__ = ['DecentralizedProtocol']

class DecentralizedProtocol(Protocol):
  """Decentralized protocol
  """

  def __init__(self, arms):
    pass


  @property
  def type(self):
    return 'decentralizedprotocol'


  def init(self):
    pass


