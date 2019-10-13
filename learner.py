"""
Abstract learner
"""

from abc import abstractmethod


class Learner():
  """Abstract class for learners"""

  @abstractmethod
  def name(self):
    pass

  def init(self, bandit, horizon):
    self._bandit = bandit
    self._horizon = horizon
    self.model_init()
    self.goal_init()
    self.learner_init()

  @abstractmethod
  def model_init(self):
    pass

  @abstractmethod
  def goal_init(self):
    pass

  @abstractmethod
  def learner_init(self):
    pass

  @abstractmethod
  def update(self):
    pass

  @abstractmethod
  def choice(self):
    pass

  @abstractmethod
  def goal(self):
    pass
