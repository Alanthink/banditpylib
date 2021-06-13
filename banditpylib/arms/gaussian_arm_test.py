from .gaussian_arm import GaussianArm


class TestGaussianArm:
  """Test Gaussian arm"""
  def test_probability_within_one_std(self):
    mu, std = 0, 1
    gaussian_arm = GaussianArm(mu, std)
    rewards = gaussian_arm.pull(1000)
    # Probability of rewards within [mu - std, mu + std]
    prob_within_one_std = [
        1 if (mu - std <= reward <= mu + std) else 0 for reward in rewards
    ]
    assert sum(prob_within_one_std) > 0.68, (
        'Probability of rewards within [-1, 1] is expected '
        'at least 0.68. Got %.2f.' % prob_within_one_std)
