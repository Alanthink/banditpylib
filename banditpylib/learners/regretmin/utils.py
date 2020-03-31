from abc import abstractmethod

from .. import Learner


class RegretMinimizationLearner(Learner):
  """Base class for a learner with goal regret minimization.

  Before a game is run, a learner should be initialized with :func:`reset`.
  During each time step, :func:`learner_step` is called to ask the learner for
   the choice of the action. :func:`update` is called by the protocol when the
  reward is revealed from the environment.
  """

  # default protocol
  protocol = 'SinglePlayerRegretMinProtocol'

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def goal(self):
    return 'Regret Minimization'

  def _goal_reset(self):
    self.__rewards = 0

  @abstractmethod
  def _model_reset(self):
    pass

  @abstractmethod
  def _learner_reset(self):
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

  def rewards(self):
    return self.__rewards

  def reset(self, bandit, stop_cond):
    # time starts from 1
    self._horizon = stop_cond
    self._bandit = bandit
    self._t = 1
    self._goal_reset()
    self._model_reset()
    self._learner_reset()

  @abstractmethod
  def learner_step(self, context):
    pass

  def update(self, context, action, feedback):
    self._goal_update(context, action, feedback)
    self._model_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1
