# -*- coding: utf-8 -*-
"""
A simple example.

To run, try `python3 example.py` under `banditpylib` root directory.
The result is output to `out/out.pdf` by default.
"""

from arm import BernoulliArm
from bandit import Bandit
from draw import draw
from learner import Uniform, UCB, MOSS
from simulator import RegretMinimizationSimulator

if __name__ == '__main__':
    MEANS = [0.3, 0.5, 0.7]
    K = len(MEANS)
    ARMS = [BernoulliArm(mean) for mean in MEANS]

    RANDOM_SEED = 0
    HORIZON = 1000
    # record regret every other `interval` times
    INTERVAL = 20
    TRIALS = 100

    BANDIT = Bandit(ARMS, RANDOM_SEED)
    LEARNERS = [Uniform(K), UCB(K, 2), MOSS(K)]
    SIMULATOR = RegretMinimizationSimulator(BANDIT, LEARNERS)

    RESULTS = SIMULATOR.sim(HORIZON, INTERVAL, TRIALS)

    draw(RESULTS, 'out/out.pdf')
