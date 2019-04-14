# -*- coding: utf-8 -*-
"""
Bandit
"""

import numpy as np

from arm import Arm


class Bandit:
    """Base bandit class"""

    def __init__(self, arms, seed):
        if not isinstance(arms, list):
            raise Exception('Arms should be given in a list!')
        for arm in arms:
            if not isinstance(arm, Arm):
                raise Exception('Not an arm!')
        self.arms = arms

        if not isinstance(seed, int):
            raise Exception('Random seed should be an integer!')
        np.random.seed(seed)

        self.arm_num = len(arms)
        if self.arm_num < 2:
            raise Exception('The number of arms should be at least two!')

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
            raise Exception('Wrong arm index!')
        return self.arms[ind].pull()

    def regret(self, pulls, rewards):
        """regret compared to the best strategy"""
        return self.best_arm.mean * pulls - rewards
