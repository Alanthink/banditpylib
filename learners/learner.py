"""
Abstract learner
"""

from abc import abstractmethod


class Learner():
  """Abstract class for learners"""

  @property
  @abstractmethod
  def name(self):
    pass

  @property
  @abstractmethod
  def goal(self):
    pass

  def init(self, bandit, horizon):
    self._bandit = bandit
    self._horizon = horizon
    self._t = 1
    self._model_init()
    self._goal_init()
    self._learner_init()

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _goal_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  def update(self, context, action, feedback):
    self._model_update(context, action, feedback)
    self._goal_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _goal_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  @abstractmethod
  def choice(self, context):
    pass
