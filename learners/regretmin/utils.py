from abc import abstractmethod

from learners import Learner

__all__ = ['RegretMinimizationLearner']


class RegretMinimizationLearner(Learner):
  """base class for regret minimization learners"""

  # pylint: disable=I0023, W0235
  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'Regret Minimization'

  def _goal_init(self):
    self.__rewards = 0

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  def _goal_update(self, context, action, feedback):
    del context, action
    self.__rewards += feedback[0]

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  @property
  def rewards(self):
    return self.__rewards

  def init(self, bandit):
    # time starts from 1
    self._bandit = bandit
    self._t = 1
    self._goal_init()
    self._model_init()
    self._learner_init()

  @abstractmethod
  def learner_choice(self, context):
    pass

  def update(self, context, action, feedback):
    self._goal_update(context, action, feedback)
    self._model_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1
