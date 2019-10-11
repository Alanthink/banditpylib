"""
Learners under the  Multinomial Logit (MNL) bandit model.
"""

from abc import abstractmethod
from absl import logging

import numpy as np

from learner import Learner
from utils import search_best_assortment


class MNLLearner(Learner):
  """Base class for learners in the MNL bandit model"""

  def init(self, bandit, horizon):
    """initialization"""
    super().init(bandit, horizon)
    if self._bandit.type != 'mnlbandit':
      logging.fatal('(mnllearner) I don\'t understand the bandit environment!')

  def local_init(self):
    self._prod_num = self._bandit.prod_num
    self._revenue = self._bandit.revenue

  @abstractmethod
  def update(self, action, feedback):
    pass

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def choice(self, time):
    pass

  @abstractmethod
  def goal(self):
    pass

  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def rewards(self):
    pass


class RegretMinimizationLearner(MNLLearner):
  """Base class for regret minimization learners"""

  def __init__(self):
    super().__init__()
    self._goal = 'Regret minimization'

  @property
  def goal(self):
    return self._goal

  @property
  def rewards(self):
    return self._rewards

  @abstractmethod
  def update(self, action, feedback):
    pass

  @abstractmethod
  def reset(self):
    self._rewards = 0

  @abstractmethod
  def choice(self, time):
    pass

  @abstractmethod
  def name(self):
    pass


class ExplorationExploitation(RegretMinimizationLearner):
  """Exploration-Exploitation algorithm for MNL-Bandit"""

  def __init__(self):
    super().__init__()
    self._name = 'Exploration-Exploitation'

  @property
  def name(self):
    return self._name

  def update(self, action, feedback):
    """
    Input
      action: a list of product indexes
      feedback: (revenue, purchase observation)
    """
    if feedback[1] == 0:
      for prod in self._s_ell:
        self._T[prod] += 1
      bar_v = self._purchases/self._T
      self._v_ucb = bar_v + np.sqrt( bar_v*48*np.log(np.sqrt(self._prod_num)*self._ell+1)/self._T) + 48*np.log(np.sqrt(self._prod_num)*self._ell+1)/self._T
      self._ell += 1
    else:
      self._purchases[feedback[1]] += 1
      self._rewards += feedback[0]

    self._t += 1

  def reset(self):
    self._ell = 1
    self._v_ucb = np.ones(self._prod_num+1)
    self._t = 1
    self._rewards = 0
    self._purchases = np.zeros(self._prod_num+1)
    self._T = np.zeros(self._prod_num+1)
    # avoid division by 0
    self._T[0] = 1
    self._update_epoch = False
    _, best_assort = search_best_assortment(self._v_ucb, self._revenue)
    self._s_ell = best_assort

  def choice(self, time):
    if self._update_epoch:
      _, best_assort = search_best_assortment(self._v_ucb, self._revenue)
      self._s_ell = best_assort
    return self._s_ell
