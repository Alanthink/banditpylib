"""
Abstract learner
"""

from abc import ABC, abstractmethod

class Learner(ABC):
  """Abstract class for learners"""

  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def rewards(self):
    pass

  @abstractmethod
  def init(self, bandit, horizon):
    self._bandit = bandit
    self._horizon = horizon
    self.local_init()
    self.reset()

  @abstractmethod
  def local_init(self):
    pass

  @abstractmethod
  def update(self, action, feedback):
    pass

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def choice(self, time):
    pass

  @abstractmethod
  def goal(self):
    pass
