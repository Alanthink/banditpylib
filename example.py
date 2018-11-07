#!/usr/bin/env python3
from arm import BernoulliArm
from bandit import Bandit
from draw import draw
from learner import Uniform, UCB
from simulator import RegretMinimizationSimulator

if __name__ == '__main__':
    means = [0.3, 0.7]
    K = len(means)
    arms = []
    for mean in means:
        arms.append(BernoulliArm(mean))

    randomSeed = 0
    bandit = Bandit(arms, randomSeed)
    learners = [Uniform(K), UCB(K, 2)]
    simulator = RegretMinimizationSimulator(bandit, learners)

    horizon = 1000
    breakpoints = []
    for i in range(horizon):
        if i > 0 and i % 10 == 0:
            breakpoints.append(i)
    trials = 100

    results = simulator.sim(horizon, breakpoints, trials)

    draw(results, 'out/out.pdf')
