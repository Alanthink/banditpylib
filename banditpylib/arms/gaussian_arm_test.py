from .gaussian_arm import GaussianArm


class TestGaussianArm:
  """Test Gaussian arm"""

  def test_probability_within_one_std(self):
    mu, var = 0, 1
    gaussian_arm = GaussianArm(mu, var)
    rewards = gaussian_arm.pull(1000)
    # rewards within one std
    prob_within_one_std = [
        1 if (mu - var <= reward <= mu + var) else 0
        for reward in rewards
    ]
    assert sum(
        prob_within_one_std
    ) > 0.68, ('Probability of rewards within one std in N(0, 1) %.2f is '
               'no greater than 0.68!' % prob_within_one_std)
