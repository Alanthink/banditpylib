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
        goal = lList[0].goal()
        for l in lList:
            if l.goal() != goal:
                log.error('Some learner has different goal!')
            if not (isinstance(l, Learner)):
                log.error('Some learner is not legimate!')
            # initialize every learner
            l.start(b.K)

        self.bandit = b
        self.learners = lList


class RegretMinimizationSimulator(Simulator):
    def __init_(self, b, lList):
        Simulator.__init__(self, b, lList)

    def sim(self, horizon, interval, trials):
        breakpoints = []
        for i in range(horizon + 1):
            if i % interval == 0:
                breakpoints.append(i)

        results = dict()
        results['breakpoints'] = breakpoints

        for l in self.learners:
            print('Simulate learner %s' % l.getName())
            totalRegret = dict()
            for trial in range(trials):
                l.reset()
                for t in range(horizon + 1):
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
