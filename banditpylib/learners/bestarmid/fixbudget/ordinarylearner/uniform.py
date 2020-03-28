import numpy as np

from .utils import OrdinaryLearner


class Uniform(OrdinaryLearner):
  """Uniform sampling policy

  Sample each arm in a round-robun fashion.
  """

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def _name(self):
    return 'Uniform'

  def _learner_reset(self):
    pass

  def learner_round(self):
    rnd_array = np.random.multinomial(
        self._budget, np.ones(self._arm_num)/self._arm_num, size=1)[0]
    action = [(ind, rnd_array[ind]) for ind in range(self._arm_num)]
    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)

  def best_arm(self):
    return max([(ind, arm.em_mean)
                for (ind, arm) in enumerate(self._em_arms)],
               key=lambda x: x[1])[0]
