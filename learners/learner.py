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
    self._t = 1
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

  def update(self, context, action, feedback):
    self.model_update(context, action, feedback)
    self.goal_update(context, action, feedback)
    self.learner_update(context, action, feedback)
    self._t += 1

  @abstractmethod
  def model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def goal_update(self, context, action, feedback):
    pass

  @abstractmethod
  def learner_update(self, context, action, feedback):
    pass

  @abstractmethod
  def choice(self, context):
    pass

  @abstractmethod
  def goal(self):
    pass
