from abc import abstractmethod

from learners import Learner

__all__ = ['RegretMinimizationLearner']


class RegretMinimizationLearner(Learner):
  """Base class for regret minimization learners"""

  @property
  @abstractmethod
  def name(self):
    pass

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

  def init(self, bandit, horizon):
    # time starts from 1
    self._bandit = bandit
    self._horizon = horizon
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

  def _agg_decentralized_regret(self):
    if not self._init:
      return 0
    else:
      return self._bandit.regret(self.__rewards)

  def _one_decentralized_iteration(self, m):
    ############################################################################
    # initialization
    if not self._init:
      self.__init()
    ############################################################################
    # simulation starts from t = 1
    context = self._bandit.context
    action = self._learner_choice(context, m)
    feedback = self._bandit.feed(action)
    self.__update(context, action, feedback)
    message = self._broadcast_message(context, action, feedback)
    return dict({self.name: message})
