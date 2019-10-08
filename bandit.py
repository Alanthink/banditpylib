"""
Bandit
"""

from absl import logging

from arm import Arm


class Bandit:
  """Base bandit class"""

  def __init__(self, arms):
    if not isinstance(arms, list):
      logging.fatal('Arms should be given in a list!')
    for arm in arms:
      if not isinstance(arm, Arm):
        logging.fatal('Not an arm!')
    self.arms = arms

    self.arm_num = len(arms)
    if self.arm_num < 2:
      logging.fatal('The number of arms should be at least two!')

    self.best_arm = self.arms[0]
    for arm in self.arms:
      if arm.mean > self.best_arm.mean:
        self.best_arm = arm

  def get_num_of_arm(self):
    """return numbe of arms"""
    return self.arm_num

  def pull(self, ind):
    """pull arm"""
    if ind not in range(self.arm_num):
      logging.fatal('Wrong arm index!')
    return self.arms[ind].pull()

  def regret(self, pulls, rewards):
    """regret compared to the best strategy"""
    return self.best_arm.mean * pulls - rewards
