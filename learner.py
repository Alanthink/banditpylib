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
    def __init__(self):
        pass

    def start(self, K):
        self.K = K
        self.emArms = [EmArm() for ind in range(self.K)]
        self.rewards = 0

    def update(self, ind, reward):
        self.emArms[ind].pulls += 1
        self.emArms[ind].rewards += reward
        self.rewards += reward

    def reset(self):
        for arm in self.emArms:
            arm.reset()
        self.rewards = 0


class RegretMinimizationLearner(Learner):
    def __init__(self):
        Learner.__init__(self)

    def goal(self):
        return "Minimize the regret"


class Uniform(RegretMinimizationLearner):
    def __init__(self):
        RegretMinimizationLearner.__init__(self)

    def choice(self, t):
        return t % self.K

    def getName(self):
        return 'Uniform'


class UCB(RegretMinimizationLearner):
    def __init__(self, alpha):
        RegretMinimizationLearner.__init__(self)
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


class MOSS(RegretMinimizationLearner):
    def __init__(self):
        RegretMinimizationLearner.__init__(self)

    def upperConfidenceBound(self, arm, t):
        return arm.getEmMean() + np.sqrt(2 / arm.pulls * np.log(max(1, t / (self.K * arm.pulls))))

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
        return 'MOSS'
