from .bernoulli_arm import BernoulliArm
from .pseudo_arm import PseudoArm


class TestPseudoArm:
  """Test pseudo-arm"""

  def test_empirical_variance(self):
    pseudo_arm = PseudoArm()
    ber_arm = BernoulliArm(0.5)
    pseudo_arm.update(ber_arm.pull(100))
    em_var = pseudo_arm.em_var
    assert em_var <= 1, \
        ('Empirical variance of Bernoulli arm %.2f is greater than 1!' % em_var)
