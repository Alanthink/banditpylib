import numpy as np


class Arm:
    def __init__(self, mean):
        self.mean = mean


class BernoulliArm(Arm):
    def __init__(self, mean):
        Arm.__init__(self, mean)

    def pull(self):
        return np.random.binomial(1, self.mean)


class GaussianArm(Arm):
    def __init__(self, mu, sigma):
        Arm.__init__(self, mu)
        self.sigma = sigma

    def pull(self):
        return np.random.normal(self.mean, self.sigma, 1)[0]
