from abc import abstractmethod

from learners.learner import Learner


class RegretMinimizationLearner(Learner):
  """Base class for regret minimization learners"""

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def choice(self, context):
    pass

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  @property
  def goal(self):
    return self.__goal

  @property
  def rewards(self):
    return self.__rewards

  def __init__(self):
    super().__init__()
    self.__goal = 'Regret minimization'

  def _goal_init(self):
    self.__rewards = 0

  def _goal_update(self, context, action, feedback):
    self.__rewards += feedback[0]
