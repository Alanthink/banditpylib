import glog as log
import numpy as np

from bandit import Bandit
from learner import Learner

class Simulator:
    def __init__(self, b, lList):
        if not (isinstance(b, Bandit)):
            log.error('Not a legimate bandit!')
        if not (type(lList) is list):
            log.error('Learners should be given in a list!')
        if len(lList) < 1:
            log.error('There should be at least one learner!')
        for l in lList:
            if not (isinstance(l, Learner)):
                log.error('Some learner is not legimate!')
            if (l.K != b.K):
                log.error('Some learner does not know the correct number of arms!')

        self.bandit = b
        self.learners = lList

    def sim(self, horizon, breakpoints, trials):
        results = dict()
        results['breakpoints'] = breakpoints

        for l in self.learners:
            totalRegret = dict()
            for trial in range(trials):
                l.reset()
                for t in range(horizon):
                    choice = l.choice(t)
                    reward = self.bandit.pull(choice)
                    l.update(choice, reward)
                    if t in breakpoints:
                        totalRegret[t] = totalRegret.get(t, 0) + \
                          self.bandit.regret(t, l.rewards)
            regret = []
            for breakpoint in breakpoints:
                regret.append(totalRegret[breakpoint] / trials)
            results[l.getName()] = regret

        return results
    