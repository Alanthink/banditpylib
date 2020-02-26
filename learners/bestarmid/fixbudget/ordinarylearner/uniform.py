from .utils import OrdinaryLearner

__all__ = ['Uniform']


class Uniform(OrdinaryLearner):
  """sample each arm the same number of times"""

  def __init__(self, pars):
    super().__init__(pars)
    self._name = self._name if self._name else 'Uniform'

  def _learner_init(self):
    pass

  def learner_run(self):
    for r in range(self._budget):
      action = r % self._arm_num
      feedback = self._bandit.feed(action)
      self._model_update(action, feedback)

  def best_arm(self):
    return max([(ind, arm.em_mean)
                for (ind, arm) in enumerate(self._em_arms)],
               key=lambda x: x[1])[0]
