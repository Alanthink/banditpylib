# -*- coding: utf-8 -*-
"""
Simulators for doing experiments
"""

from bandit import Bandit
from learner import Learner


class Simulator:  # pylint: disable=too-few-public-methods
    """Base class for simulator"""

    def __init__(self, bandit, learners):
        if not isinstance(bandit, Bandit):
            raise Exception('Not a legimate bandit!')
        if not isinstance(learners, list):
            raise Exception('Learners should be given in a list!')
        if len(learners) < 1:
            raise Exception('There should be at least one learner!')
        goal = learners[0].goal()
        for learner in learners:
            if learner.goal() != goal:
                raise Exception('Some learner has different goal!')
            if not isinstance(learner, Learner):
                raise Exception('Some learner is not legimate!')

        self.bandit = bandit
        self.learners = learners


class RegretMinimizationSimulator(Simulator):
    # pylint: disable=too-few-public-methods
    """Simulator for regret minimization """

    def __init_(self, bandit, learners):
        Simulator.__init__(self, bandit, learners)

    def sim(self, horizon, interval, trials):
        # pylint: disable=too-many-locals
        """Simulation method"""
        breakpoints = []
        for i in range(horizon + 1):
            if i % interval == 0:
                breakpoints.append(i)

        results = dict()
        results['breakpoints'] = breakpoints

        for learner in self.learners:
            print('Simulate learner %s' % learner.get_name())
            total_regret = dict()
            for trial in range(trials):  # pylint: disable=unused-variable
                learner.reset()
                for time in range(horizon + 1):
                    choice = learner.choice(time)
                    reward = self.bandit.pull(choice)
                    learner.update(choice, reward)
                    if time in breakpoints:
                        total_regret[time] = total_regret.get(time, 0) + \
                          self.bandit.regret(time, learner.rewards)
            regret = []
            for breakpoint in breakpoints:
                regret.append(total_regret[breakpoint] / trials)
            results[learner.get_name()] = regret

        return results
