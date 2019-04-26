# -*- coding: utf-8 -*-
"""
A simple example.

To run, try `python3 example.py` under `banditpylib` root directory.
The result is output to `out/out.pdf` by default.
"""

from arm import BernoulliArm
from bandit import Bandit
from draw import draw
from learner import Uniform, UCB, MOSS, TS
from simulator import RegretMinimizationSimulator

if __name__ == '__main__':
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    # make sure to get the same result for each run
    random_seed = 0
    bandit = Bandit(arms, random_seed)
    learners = [Uniform(), UCB(2), MOSS(), TS()]
    simulator = RegretMinimizationSimulator(bandit, learners)

    horizon = 2000
    # record regret every `gap` times
    gap = 20
    trials = 100

    results = simulator.sim(horizon, gap, trials)
    draw(results, 'out/out.pdf')
