import glog as log
import numpy as np


class EmArm:
    def __init__(self):
        self.pulls = 0
        self.rewards = 0

    def getEmMean(self):
        if self.pulls == 0:
            log.error('No empirical mean yet!')
        return self.rewards / self.pulls

    def reset(self):
        self.pulls = 0
        self.rewards = 0


class Learner:
    def __init__(self, K):
        self.K = K
        self.emArms = []
        for ind in range(self.K):
            self.emArms.append(EmArm())
        self.rewards = 0

    def update(self, ind, reward):
        self.emArms[ind].pulls += 1
        self.emArms[ind].rewards += reward
        self.rewards += reward

    def reset(self):
        for arm in self.emArms:
            arm.reset()
        self.rewards = 0


class Uniform(Learner):
    def __init__(self, K):
        Learner.__init__(self, K)

    def choice(self, t):
        return t % self.K

    def getName(self):
        return 'Uniform'


class UCB(Learner):
    def __init__(self, K, alpha):
        Learner.__init__(self, K)
        self.alpha = alpha

    def upperConfidenceBound(self, arm, t):
        return arm.getEmMean() + np.sqrt(self.alpha / arm.pulls * np.log(t))

    def choice(self, t):
        if (t < self.K):
            return t % self.K

        armToChoose = 0
        for ind in range(self.K):
            if self.upperConfidenceBound(self.emArms[ind], t) > \
              self.upperConfidenceBound(self.emArms[armToChoose], t):
                armToChoose = ind
        return armToChoose

    def getName(self):
        return 'UCB'   
