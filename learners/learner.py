"""
Abstract learner
"""

from abc import ABC, abstractmethod


class Learner(ABC):
  """Abstract class for learners"""

  # learner goal
  @property
  @abstractmethod
  def goal(self):
    pass

  # learner name
  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _goal_init(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def _goal_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  # action suggested by the learner
  @abstractmethod
  def choice(self, context):
    pass

  def __init__(self):
    pass

  def init(self, bandit, horizon):
    # time starts from 1
    self._bandit = bandit
    self.__horizon = horizon
    self._t = 1
    self._goal_init()
    self._model_init()
    self._learner_init()

  def update(self, context, action, feedback):
    self._goal_update(context, action, feedback)
    self._model_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1

  @property
  def horizon(self):
    return self.__horizon
