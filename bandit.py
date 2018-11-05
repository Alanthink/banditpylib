from arm import Arm

import glog as log
import numpy as np


class Bandit:
    def __init__(self, arms, seed):
        if not (type(arms) is list):
            log.error('Arms should be given in a list!')
        for arm in arms:
            if not isinstance(arm, Arm):
                log.error('Not an arm!')
        self.arms = arms

        if not (type(seed) is int):
            log.error('Random seed should be an integer!')
        np.random.seed(seed)

        self.K = len(arms)
        if self.K < 2:
            log.error('The number of arms should be at least two!')
        
        self.bestArm = self.arms[0]
        for arm in self.arms:
            if arm.mean > self.bestArm.mean:
                self.bestArm = arm

    def pull(self, ind):
        if ind not in range(self.K):
            log.error('Wrong arm index!')
        return self.arms[ind].pull()


class RegretBandit(Bandit):
    """Regret minimization"""
    def __init__(self, arms, seed):
        Bandit.__init__(self, arms, seed)

    def regret(self, pulls, rewards):
        return self.bestArm.mean * pulls - rewards
