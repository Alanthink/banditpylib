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

  def init(self, bandit):
    """initialization"""
    if bandit.get_type() != 'mnlbandit':
      logging.fatal('(mnllearner) I don\'t understand the bandit environment!')
    self.prod_num = bandit.get_prod_num()
    self.revenue = bandit.get_revenue()
    self.reset()

  @abstractmethod
  def update(self, action, feedback):
    pass

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def choice(self, time):
    pass


class RegretMinimizationLearner(MNLLearner):
  """Base class for regret minimization learners"""

  def __init__(self):
    super().__init__()
    self.goal = 'Regret minimization'

  @abstractmethod
  def update(self, action, feedback):
    pass

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def choice(self, time):
    pass


class ExplorationExploitation(RegretMinimizationLearner):
  """Exploration-Exploitation algorithm for MNL-Bandit"""

  def __init__(self):
    super().__init__()
    self.name = 'Exploration-Exploitation'

  def update(self, action, feedback):
    """
    Input
      action: a list of product indexes
      feedback: (revenue, purchase observation)
    """
    if feedback[1] == 0:
      for prod in self.s_ell:
        self.T[prod] += 1
      bar_v = self.purchases/self.T
      self.v_ucb = bar_v + np.sqrt( bar_v*48*np.log(np.sqrt(self.prod_num)*self.ell+1)/self.T) + 48*np.log(np.sqrt(self.prod_num)*self.ell+1)/self.T
      self.ell += 1
    else:
      self.purchases[feedback[1]] += 1
      self.rewards += feedback[0]

    self.t += 1

  def reset(self):
    self.ell = 1
    self.v_ucb = np.ones(self.prod_num+1)
    self.t = 1
    self.rewards = 0
    self.purchases = np.zeros(self.prod_num+1)
    self.T = np.zeros(self.prod_num+1)
    # avoid division by 0
    self.T[0] = 1
    self.update_epoch = False
    _, best_assort = search_best_assortment(self.v_ucb, self.revenue)
    self.s_ell = best_assort

  def choice(self, time):
    if self.update_epoch:
      _, best_assort = search_best_assortment(self.v_ucb, self.revenue)
      self.s_ell = best_assort
    return self.s_ell
