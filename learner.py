# -*- coding: utf-8 -*-
"""
Different strategies
"""

import numpy as np


class EmArm:
    """Data Structure for storing empirical information of each arm"""

    def __init__(self):
        self.pulls = 0
        self.rewards = 0

    def get_em_mean(self):
        """get empirical mean"""
        if self.pulls == 0:
            raise Exception('No empirical mean yet!')
        return self.rewards / self.pulls

    def reset(self):
        """clear historical records"""
        self.pulls = 0
        self.rewards = 0


class Learner:
    """Base class for learners"""

    def __init__(self):
        pass

    def init(self, K):
        """initialize K arms"""
        self.arm_num = K
        self.em_arms = [EmArm() for ind in range(self.arm_num)]
        self.rewards = 0

    def update(self, ind, reward):
        """update historical record for a specific arm"""
        self.em_arms[ind].pulls += 1
        self.em_arms[ind].rewards += reward
        self.rewards += reward

    def reset(self):
        """clear historical records for all arms"""
        for arm in self.em_arms:
            arm.reset()
        self.rewards = 0


class RegretMinimizationLearner(Learner):
    """Base class for regret minimization learners"""

    def __init__(self):
        Learner.__init__(self)

    @classmethod
    def goal(cls):
        """return goal of this kind of learners"""
        return "Minimize the regret"


class Uniform(RegretMinimizationLearner):
    """Naive uniform algorithm: sample each arm the same number of times"""

    def __init__(self):
        RegretMinimizationLearner.__init__(self)

    def choice(self, time):
        """return an arm to pull"""
        return time % self.arm_num

    def get_name(cls):
        """return name of the learner"""
        return 'Uniform'


class UCB(RegretMinimizationLearner):
    """UCB"""

    def __init__(self, alpha):
        RegretMinimizationLearner.__init__(self)
        self.alpha = alpha

    def upper_confidence_bound(self, arm, time):
        """upper confidence bound"""
        return arm.get_em_mean() + \
            np.sqrt(self.alpha / arm.pulls * np.log(time))

    def choice(self, time):
        """return an arm to pull"""
        if time < self.arm_num:
            return time % self.arm_num

        arm_to_pull = 0
        for ind in range(self.arm_num):
            if self.upper_confidence_bound(self.em_arms[ind], time) > \
              self.upper_confidence_bound(self.em_arms[arm_to_pull], time):
                arm_to_pull = ind
        return arm_to_pull

    def get_name(cls):
        """return name of the learner"""
        return 'UCB'


class MOSS(RegretMinimizationLearner):
    """MOSS"""

    def __init__(self):
        RegretMinimizationLearner.__init__(self)

    def upper_confidence_bound(self, arm, time):
        """upper confidence bound"""
        return arm.get_em_mean() + np.sqrt(
            2 / arm.pulls * np.log(max(1, time / (self.arm_num * arm.pulls))))

    def choice(self, time):
        """return an arm to pull"""
        if time < self.arm_num:
            return time % self.arm_num

        arm_to_pull = 0
        for ind in range(self.arm_num):
            if self.upper_confidence_bound(self.em_arms[ind], time) > \
              self.upper_confidence_bound(self.em_arms[arm_to_pull], time):
                arm_to_pull = ind
        return arm_to_pull

    def get_name(cls):
        """return name of the learner"""
        return 'MOSS'
