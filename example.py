#!/usr/bin/env python3
from arm import BernoulliArm
from bandit import Bandit
from draw import draw
from learner import Uniform, UCB, MOSS
from simulator import RegretMinimizationSimulator

if __name__ == '__main__':
    means = [0.3, 0.5, 0.7]
    K = len(means)
    arms = [BernoulliArm(mean) for mean in means]

    randomSeed = 0
    bandit = Bandit(arms, randomSeed)
    learners = [Uniform(), UCB(2), MOSS()]
    simulator = RegretMinimizationSimulator(bandit, learners)

    horizon = 1000
    # record regret every other `interval` times
    interval = 20
    trials = 100

    results = simulator.sim(horizon, interval, trials)

    draw(results, 'out/out.pdf')
